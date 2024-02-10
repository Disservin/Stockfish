/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

#ifndef HISTORY_H_INCLUDED
#define HISTORY_H_INCLUDED

#include <array>
#include <cmath>
#include <limits>
#include <algorithm>

#include "types.h"
#include "position.h"

namespace Stockfish {

namespace Search {
struct Stack;
class Worker;
}

// StatsEntry stores the stat table value. It is usually a number but could
// be a move or even a nested history. We use a class instead of a naked value
// to directly call history update operator<<() on the entry so to use stats
// tables at caller sites as simple multi-dim arrays.
template<typename T, int D>
class StatsEntry {

    T entry;

   public:
    void operator=(const T& v) { entry = v; }
    T*   operator&() { return &entry; }
    T*   operator->() { return &entry; }
    operator const T&() const { return entry; }

    void operator<<(int bonus) {
        assert(std::abs(bonus) <= D);  // Ensure range is [-D, D]
        static_assert(D <= std::numeric_limits<T>::max(), "D overflows T");

        entry += bonus - entry * std::abs(bonus) / D;

        assert(std::abs(entry) <= D);
    }
};

// Stats is a generic N-dimensional array used to store various statistics.
// The first template parameter T is the base type of the array, and the second
// template parameter D limits the range of updates in [-D, D] when we update
// values with the << operator, while the last parameters (Size and Sizes)
// encode the dimensions of the array.
template<typename T, int D, int Size, int... Sizes>
struct Stats: public std::array<Stats<T, D, Sizes...>, Size> {
    using stats = Stats<T, D, Size, Sizes...>;

    void fill(const T& v) {

        // For standard-layout 'this' points to the first struct member
        assert(std::is_standard_layout_v<stats>);

        using entry = StatsEntry<T, D>;
        entry* p    = reinterpret_cast<entry*>(this);
        std::fill(p, p + sizeof(*this) / sizeof(entry), v);
    }
};

template<typename T, int D, int Size>
struct Stats<T, D, Size>: public std::array<StatsEntry<T, D>, Size> {};

// In stats table, D=0 means that the template parameter is not used
enum StatsParams {
    NOT_USED = 0
};

enum StatsType {
    NoCaptures,
    Captures
};

constexpr int PAWN_HISTORY_SIZE        = 512;    // has to be a power of 2
constexpr int CORRECTION_HISTORY_SIZE  = 16384;  // has to be a power of 2
constexpr int CORRECTION_HISTORY_LIMIT = 1024;

static_assert((PAWN_HISTORY_SIZE & (PAWN_HISTORY_SIZE - 1)) == 0,
              "PAWN_HISTORY_SIZE has to be a power of 2");

static_assert((CORRECTION_HISTORY_SIZE & (CORRECTION_HISTORY_SIZE - 1)) == 0,
              "CORRECTION_HISTORY_SIZE has to be a power of 2");


// ButterflyHistory records how often quiet moves have been successful or unsuccessful
// during the current search, and is used for reduction and move ordering decisions.
// It uses 2 tables (one for each color) indexed by the move's from and to squares,
// see www.chessprogramming.org/Butterfly_Boards (~11 elo)
using ButterflyHistory = Stats<int16_t, 7183, COLOR_NB, int(SQUARE_NB) * int(SQUARE_NB)>;

// CounterMoveHistory stores counter moves indexed by [piece][to] of the previous
// move, see www.chessprogramming.org/Countermove_Heuristic
using CounterMoveHistory = Stats<Move, NOT_USED, PIECE_NB, SQUARE_NB>;

// CapturePieceToHistory is addressed by a move's [piece][to][captured piece type]
using CapturePieceToHistory = Stats<int16_t, 10692, PIECE_NB, SQUARE_NB, PIECE_TYPE_NB>;

// PieceToHistory is like ButterflyHistory but is addressed by a move's [piece][to]
using PieceToHistory = Stats<int16_t, 29952, PIECE_NB, SQUARE_NB>;

// ContinuationHistory is the combined history of a given pair of moves, usually
// the current one given a previous one. The nested history table is based on
// PieceToHistory instead of ButterflyBoards.
// (~63 elo)
using ContinuationHistory = Stats<PieceToHistory, NOT_USED, PIECE_NB, SQUARE_NB>;

// PawnHistory is addressed by the pawn structure and a move's [piece][to]
using PawnHistory = Stats<int16_t, 8192, PAWN_HISTORY_SIZE, PIECE_NB, SQUARE_NB>;

// CorrectionHistory is addressed by color and pawn structure
using CorrectionHistory =
  Stats<int16_t, CORRECTION_HISTORY_LIMIT, COLOR_NB, CORRECTION_HISTORY_SIZE>;

enum PawnHistoryType {
    Normal,
    Correction
};

template<PawnHistoryType T = Normal>
int pawn_structure_index(const Position& pos) {
    return pos.pawn_key() & ((T == Normal ? PAWN_HISTORY_SIZE : CORRECTION_HISTORY_SIZE) - 1);
}

// History and stats update bonus, based on depth
int stat_bonus(Depth d);

// History and stats update malus, based on depth
int stat_malus(Depth d);

// Updates histories of the move pairs formed
// by moves at ply -1, -2, -3, -4, and -6 with current move.
void update_continuation_histories(Search::Stack* ss, Piece pc, Square to, int bonus);

// Updates move sorting heuristics
void update_quiet_stats(
  const Position& pos, Search::Stack* ss, Search::Worker& workerThread, Move move, int bonus);

// Updates stats at the end of search() when a bestMove is found
void update_all_stats(const Position& pos,
                      Search::Stack*  ss,
                      Search::Worker& workerThread,
                      Move            bestMove,
                      Value           bestValue,
                      Value           beta,
                      Square          prevSq,
                      Move*           quietsSearched,
                      int             quietCount,
                      Move*           capturesSearched,
                      int             captureCount,
                      Depth           depth);
}

#endif  // HISTORY_H_INCLUDED
