/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "movepick.h"

#include <cassert>
#include <limits>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "position.h"

namespace Stockfish {

namespace {

enum Stages {
    // generate main search moves
    MAIN_TT,
    CAPTURE_INIT,
    GOOD_CAPTURE,
    QUIET_INIT,
    GOOD_QUIET,
    BAD_CAPTURE,
    BAD_QUIET,

    // generate evasion moves
    EVASION_TT,
    EVASION_INIT,
    EVASION,

    // generate probcut moves
    PROBCUT_TT,
    PROBCUT_INIT,
    PROBCUT,

    // generate qsearch moves
    QSEARCH_TT,
    QCAPTURE_INIT,
    QCAPTURE
};

#ifdef USE_AVX512

alignas(16) inline constexpr std::array<std::uint8_t, 16> IotaBytes = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

alignas(16) inline constexpr std::array<std::uint8_t, 16> PackLoBytes = {
  0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};

alignas(16) inline constexpr std::array<std::uint8_t, 16> PackHiBytes = {
  8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

inline __m512i expand_indices(const std::array<std::uint8_t, 16>& bytes) noexcept {
    return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes.data())));
}

// Maintains up to 16 moves in descending score order.
//
// Moves and scores are kept in separate 16-lane vectors. To insert a move,
// all scores are compared in parallel to find the insertion point. The
// resulting mask is converted into permutation indices that shift every lane
// after that point one position to the right, leaving one lane for the new
// move and score.
//
// Unused score lanes contain INT_MIN, which guarantees an insertion point for
// every valid score. When writing the result, the move and score vectors are
// interleaved again to reconstruct ExtMove objects.
struct MoveSorter {
    static constexpr isize MAX_ELEMENTS = 16;

    __m512i sortedValues;
    __m512i sortedMoves;
    __m512i iota;

    explicit MoveSorter(const ExtMove& first) {
        sortedMoves  = _mm512_set1_epi32(first.raw());
        sortedValues = _mm512_set1_epi32(first.value);

        iota = expand_indices(IotaBytes);

        // Set the uninitialized move values to INT_MIN, so that they sort less than any other move
        sortedValues = _mm512_mask_set1_epi32(sortedValues, static_cast<__mmask16>(0xFFFE),
                                              std::numeric_limits<int>::min());
    }

    void insert(const ExtMove& m) {
        assert(m.value != std::numeric_limits<int>::min());

        const __m512i value = _mm512_set1_epi32(m.value);
        const __m512i move  = _mm512_set1_epi32(m.raw());

        const __mmask16 beyond = _mm512_cmplt_epi32_mask(sortedValues, value);

        // INT_MIN sentinels guarantee at least one bit is set.
        assert(beyond != 0);

        // If beyond = 11111000...
        // subtracting 1 gives 11110111...
        // i.e. every lane except the insertion lane.
        const __mmask16 allButInsertion = _kadd_mask16(beyond, static_cast<__mmask16>(0xFFFF));

        // movm gives -1 for set mask bits and 0 otherwise.
        //
        // index:
        //   before insertion -> iota
        //   insertion onward -> iota - 1
        const __m512i index = _mm512_add_epi32(iota, _mm512_movm_epi32(beyond));

        sortedValues = _mm512_mask_permutexvar_epi32(value, allButInsertion, index, sortedValues);

        sortedMoves = _mm512_mask_permutexvar_epi32(move, allButInsertion, index, sortedMoves);
    }

    void write_sorted(ExtMove* moves, isize count) const {
        static_assert(sizeof(ExtMove) == 8);

        assert(count >= 1);
        assert(count <= MAX_ELEMENTS);

        //
        // First 8 ExtMoves
        //
        const __m512i loIndices = expand_indices(PackLoBytes);

        const __m512i lo = _mm512_permutex2var_epi32(sortedMoves, loIndices, sortedValues);

        // count is <= 16. Truncation to __mmask8 intentionally
        // gives 0xFF when count >= 8.
        const __mmask8 loMask = static_cast<__mmask8>((1u << count) - 1u);

        _mm512_mask_storeu_epi64(moves, loMask, lo);

        if (count <= 8)
            return;

        const __m512i hiIndices = expand_indices(PackHiBytes);

        const __m512i hi = _mm512_permutex2var_epi32(sortedMoves, hiIndices, sortedValues);

        const __mmask8 hiMask = static_cast<__mmask8>((1u << (count - 8)) - 1u);

        _mm512_mask_storeu_epi64(moves + 8, hiMask, hi);
    }
};
#endif

// Sort moves in descending order up to and including a given limit.
// The order of moves smaller than the limit is left unspecified.
void partial_insertion_sort(ExtMove* begin, ExtMove* end, int limit) {
    ExtMove* sortedEnd = begin;
    ExtMove* p;

#ifdef USE_AVX512

    if (begin == end)
        return;

    p = begin + 1;

    MoveSorter sorter(*begin);

    isize count = 1;

    for (; p != end; ++p)
    {
        if (p->value < limit)
            continue;

        if (count == MoveSorter::MAX_ELEMENTS)
            break;

        sorter.insert(*p);

        *p = begin[count];
        ++count;
    }

    sorter.write_sorted(begin, count);

    sortedEnd = begin + count - 1;

#else

    p = begin + 1;

#endif

    // Scalar implementation.
    for (; p != end; ++p)
    {
        if (p->value < limit)
            continue;

        ExtMove tmp = *p;

        *p = *++sortedEnd;

        ExtMove* q = sortedEnd;

        while (q != begin && *(q - 1) < tmp)
        {
            *q = *(q - 1);
            --q;
        }

        *q = tmp;
    }
}

}  // namespace


// Constructors of the MovePicker class. As arguments, we pass information
// to decide which class of moves to emit, to help sorting the (presumably)
// good moves first, and how important move ordering is at the current node.

// MovePicker constructor for the main search and for the quiescence search
MovePicker::MovePicker(const Position&              p,
                       Move                         ttm,
                       Depth                        d,
                       const ButterflyHistory*      mh,
                       const LowPlyHistory*         lph,
                       const CapturePieceToHistory* cph,
                       const PieceToHistory**       ch,
                       const SharedHistories*       sh,
                       int                          pl) :
    pos(p),
    mainHistory(mh),
    lowPlyHistory(lph),
    captureHistory(cph),
    continuationHistory(ch),
    sharedHistory(sh),
    ttMove(ttm),
    depth(d),
    ply(pl) {

    if (pos.checkers())
        stage = EVASION_TT + !(ttm && pos.pseudo_legal(ttm));

    else
        stage = (depth > 0 ? MAIN_TT : QSEARCH_TT) + !(ttm && pos.pseudo_legal(ttm));
}

// MovePicker constructor for ProbCut: we generate captures with Static Exchange
// Evaluation (SEE) greater than or equal to the given threshold.
MovePicker::MovePicker(const Position& p, Move ttm, int th, const CapturePieceToHistory* cph) :
    pos(p),
    captureHistory(cph),
    ttMove(ttm),
    threshold(th) {
    assert(!pos.checkers());

    stage = PROBCUT_TT + !(ttm && pos.capture_stage(ttm) && pos.pseudo_legal(ttm));
}

// Assigns a numerical value to each move in a list, used for sorting.
// Captures are ordered by Most Valuable Victim (MVV), preferring captures
// with a good history. Quiet moves are ordered using the history tables.
template<GenType Type>
ExtMove* MovePicker::score(const MoveList<Type>& ml) {

    static_assert(Type == CAPTURES || Type == QUIETS || Type == EVASIONS, "Wrong type");

    Color us = pos.side_to_move();

    [[maybe_unused]] Bitboard threatByLesser[KING + 1];
    if constexpr (Type == QUIETS)
    {
        threatByLesser[PAWN]   = 0;
        threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
        threatByLesser[ROOK] =
          pos.attacks_by<KNIGHT>(~us) | pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
        threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
        threatByLesser[KING]  = 0;
    }

    ExtMove* it = cur;
    for (auto move : ml)
    {
        ExtMove& m = *it++;
        m          = move;

        const Square    from          = m.from_sq();
        const Square    to            = m.to_sq();
        const Piece     pc            = pos.moved_piece(m);
        const PieceType pt            = type_of(pc);
        const Piece     capturedPiece = pos.piece_on(to);

        if constexpr (Type == CAPTURES)
            m.value = (*captureHistory)[pc][to][type_of(capturedPiece)]
                    + 7 * int(PieceValue[capturedPiece]);

        else if constexpr (Type == QUIETS)
        {
            // histories
            m.value = 2 * (*mainHistory)[us][m.raw()];
            m.value += 2 * sharedHistory->pawn_entry(pos)[pc][to];
            m.value += (*continuationHistory[0])[pc][to];
            m.value += (*continuationHistory[1])[pc][to];
            m.value += (*continuationHistory[2])[pc][to];
            m.value += (*continuationHistory[3])[pc][to];
            m.value += (*continuationHistory[5])[pc][to];

            // bonus for checks
            m.value += ((pos.check_squares(pt) & to) && pos.see_ge(m, -75)) * 16384;

            // penalty for moving to a square threatened by a lesser piece
            // or bonus for escaping an attack by a lesser piece.
            int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
            m.value += PieceValue[pt] * v;


            if (ply < LOW_PLY_HISTORY_SIZE)
                m.value += 8 * (*lowPlyHistory)[ply][m.raw()] / (1 + ply);
        }

        else  // Type == EVASIONS
        {
            if (pos.capture_stage(m))
                m.value = PieceValue[capturedPiece] + (1 << 28);
            else
                m.value = (*mainHistory)[us][m.raw()] + (*continuationHistory[0])[pc][to];
        }
    }
    return it;
}

// Returns the next move satisfying a predicate function.
// This never returns the TT move, as it was emitted before.
template<typename Pred>
Move MovePicker::select(Pred filter) {

    for (; cur < endCur; ++cur)
        if (*cur != ttMove && filter())
            return *cur++;

    return Move::none();
}

// This is the most important method of the MovePicker class. We emit one
// new pseudo-legal move on every call until there are no more moves left,
// picking the move with the highest score from a list of generated moves.
Move MovePicker::next_move() {

    constexpr int goodQuietThreshold = -14000;
top:
    switch (stage)
    {

    case MAIN_TT :
    case EVASION_TT :
    case QSEARCH_TT :
    case PROBCUT_TT :
        ++stage;
        return ttMove;

    case CAPTURE_INIT :
    case PROBCUT_INIT :
    case QCAPTURE_INIT : {
        MoveList<CAPTURES> ml(pos);

        cur = endBadCaptures = moves;
        endCur = endCaptures = score<CAPTURES>(ml);

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
        ++stage;
        goto top;
    }

    case GOOD_CAPTURE :
        if (select([&]() {
                if (pos.see_ge(*cur, -cur->value / 18))
                    return true;
                std::swap(*endBadCaptures++, *cur);
                return false;
            }))
            return *(cur - 1);

        ++stage;
        [[fallthrough]];

    case QUIET_INIT :
        if (!skipQuiets)
        {
            MoveList<QUIETS> ml(pos);

            endCur = endGenerated = score<QUIETS>(ml);

            partial_insertion_sort(cur, endCur, -3560 * depth);
        }

        ++stage;
        [[fallthrough]];

    case GOOD_QUIET :
        if (!skipQuiets && select([&]() { return cur->value > goodQuietThreshold; }))
            return *(cur - 1);

        // Prepare the pointers to loop over the bad captures
        cur    = moves;
        endCur = endBadCaptures;

        ++stage;
        [[fallthrough]];

    case BAD_CAPTURE :
        if (select([]() { return true; }))
            return *(cur - 1);

        // Prepare the pointers to loop over quiets again
        cur    = endCaptures;
        endCur = endGenerated;

        ++stage;
        [[fallthrough]];

    case BAD_QUIET :
        if (!skipQuiets)
            return select([&]() { return cur->value <= goodQuietThreshold; });

        return Move::none();

    case EVASION_INIT : {
        MoveList<EVASIONS> ml(pos);

        cur    = moves;
        endCur = endGenerated = score<EVASIONS>(ml);

        partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
        ++stage;
        [[fallthrough]];
    }

    case EVASION :
    case QCAPTURE :
        return select([]() { return true; });

    case PROBCUT :
        return select([&]() { return pos.see_ge(*cur, threshold); });
    }

    assert(false);
    return Move::none();  // Silence warning
}

void MovePicker::skip_quiet_moves() { skipQuiets = true; }

}  // namespace Stockfish