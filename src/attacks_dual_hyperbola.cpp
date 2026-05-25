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

#include "attacks_dual_hyperbola.h"

#ifdef USE_DUAL_HYPERBOLA_QUINT

    #include <array>
    #include <immintrin.h>

namespace Stockfish::Attacks {
namespace {

alignas(64) DualMagic DualMagics[SQUARE_NB];

// Sliding attacks within a rank, indexed by the slider's file and the
// 8-bit rank occupancy, yielding the 8-bit attack set on that rank.
constexpr auto RankAttacks = []() {
    std::array<std::array<uint8_t, 256>, FILE_NB> table{};
    for (int file = 0; file < 8; ++file)
        for (int occ = 0; occ < 256; ++occ)
        {
            uint8_t attacks = 0;
            for (int f = file + 1; f <= 7; ++f)
            {
                attacks |= uint8_t(1 << f);
                if (occ & (1 << f))
                    break;
            }
            for (int f = file - 1; f >= 0; --f)
            {
                attacks |= uint8_t(1 << f);
                if (occ & (1 << f))
                    break;
            }
            table[file][occ] = attacks;
        }
    return table;
}();

std::pair<Bitboard, Bitboard> attacks_bb(const DualMagic& m, Bitboard occupied) {
    const auto bswap = [](__m256i v) {
        return _mm256_shuffle_epi8(v, _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                      13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                      10, 11, 12, 13, 14, 15));
    };

    const __m256i mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(&m));
    const __m256i rs   = _mm256_set1_epi64x(m.r);
    const __m256i rrs  = _mm256_set1_epi64x(m.rr);

    __m256i o      = _mm256_and_si256(mask, _mm256_set1_epi64x(occupied));
    __m256i fwd    = _mm256_sub_epi64(o, rs);
    __m256i rev    = bswap(_mm256_sub_epi64(bswap(o), rrs));
    __m256i result = _mm256_and_si256(_mm256_xor_si256(fwd, rev), mask);

    __m128i rookBishop =
      _mm_or_si128(_mm256_extracti128_si256(result, 1), _mm256_castsi256_si128(result));

    Bitboard rowOccupancy = m.rankAttacksLookup[(occupied >> m.shift) & 0xff];
    Bitboard rankAttacks  = rowOccupancy << m.shift;

    return {_mm_extract_epi64(rookBishop, 1), _mm_cvtsi128_si64(rookBishop) + rankAttacks};
}

}  // namespace

void init_impl() {
    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        DualMagic& m        = DualMagics[s];
        m.maskFile          = line_mask(s, NORTH, SOUTH);
        m.maskDiag          = line_mask(s, NORTH_EAST, SOUTH_WEST);
        m.maskNone          = 0;
        m.maskAntidiag      = line_mask(s, NORTH_WEST, SOUTH_EAST);
        m.r                 = square_bb(s) * 2;
        m.rr                = square_bb(Square(63 - int(s))) * 2;
        m.rankAttacksLookup = RankAttacks[int(file_of(s))].data();
        m.shift             = 8 * int(rank_of(s));
    }
}

std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied) {
    assert(is_ok(s));
    return attacks_bb(DualMagics[s], occupied);
}

Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    return both_attacks_bb(s, occupied).first;
}

Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    return both_attacks_bb(s, occupied).second;
}

}  // namespace Stockfish::Attacks

#endif
