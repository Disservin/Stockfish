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

#include "attacks.h"

#if !defined(USE_HYPERBOLA_QUINT) && !defined(USE_DUAL_HYPERBOLA_QUINT)

    #include <array>

    #include "misc.h"

namespace Stockfish::Attacks {
namespace {

#ifdef USE_PEXT
using MagicMask = uint16_t;
#else
using MagicMask = Bitboard;
#endif

alignas(64) Magic Magics[SQUARE_NB][2];

[[maybe_unused]] constexpr Bitboard constexpr_pext(Bitboard b, Bitboard m) {
    Bitboard result = 0, bit = 0;
    while (m)
    {
        Bitboard last = m & -m;
        result |= bool(b & last) << bit++;
        m ^= last;
    }
    return result;
}

    #ifdef USE_COMPTIME_ATTACKS
constexpr
    #endif
  void
  init_magics(PieceType pt, MagicMask table[], Magic magics[][2], [[maybe_unused]] bool tableAlreadyInit) {
    #if !defined(USE_COMPTIME_ATTACKS)
    tableAlreadyInit = false;
    #endif

    #ifndef USE_PEXT
    int seeds[][RANK_NB] = {{8977, 44560, 54343, 38998, 5731, 95205, 104912, 17020},
                            {728, 10316, 55013, 32803, 12281, 15100, 16645, 255}};

    Bitboard occupancy[4096];
    int      epoch[4096] = {}, cnt = 0;
    Bitboard reference[4096] = {};
    #endif

    int size = 0;

    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        Bitboard edges = ((Rank1BB | Rank8BB) & ~rank_bb(s)) | ((FileABB | FileHBB) & ~file_bb(s));

        Magic&   m       = magics[s][pt - BISHOP];
        Bitboard attacks = sliding_attack(pt, s, 0);
        m.mask           = attacks & ~edges;
    #ifdef USE_PEXT
        m.pseudoAttacks = attacks;
    #else
        m.shift = (Is64Bit ? 64 : 32) - popcount(m.mask);
    #endif
        m.attacks = s == SQ_A1 ? table : magics[s - 1][pt - BISHOP].attacks + size;
        size      = 0;

        Bitboard                  b           = 0;
        [[maybe_unused]] Bitboard prevSliding = -1;
        do
        {
    #ifdef USE_PEXT
            if (!tableAlreadyInit)
            {
                Bitboard sliding = sliding_attack(pt, s, b);
                m.attacks[size] =
                  sliding != prevSliding ? constexpr_pext(sliding, attacks) : m.attacks[size - 1];
                prevSliding = sliding;
            }
    #else
            occupancy[size] = b;
            reference[size] = sliding_attack(pt, s, b);
    #endif

            size++;
            b = (b - m.mask) & m.mask;
        } while (b);

    #ifndef USE_PEXT
        PRNG rng(seeds[Is64Bit][rank_of(s)]);

        for (int i = 0; i < size;)
        {
            for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;)
                m.magic = rng.sparse_rand<Bitboard>();

            for (++cnt, i = 0; i < size; ++i)
            {
                unsigned idx = m.index(occupancy[i]);

                if (epoch[idx] < cnt)
                {
                    epoch[idx]     = cnt;
                    m.attacks[idx] = reference[i];
                }
                else if (m.attacks[idx] != reference[i])
                    break;
            }
        }
    #endif
    }
}

    #if defined(USE_COMPTIME_ATTACKS) && defined(USE_PEXT)
constexpr auto RookTable = []() {
    std::array<uint16_t, 0x19000> result{};
    Magic                         magics[64][2] = {};
    init_magics(ROOK, result.data(), magics, false);
    return result;
}();
constexpr auto BishopTable = []() {
    std::array<uint16_t, 0x1480> result{};
    Magic                        magics[64][2] = {};
    init_magics(BISHOP, result.data(), magics, false);
    return result;
}();
    #else
std::array<MagicMask, 0x19000> RookTable;
std::array<MagicMask, 0x1480>  BishopTable;
    #endif

}  // namespace

void init_impl() {
    init_magics(ROOK, const_cast<MagicMask*>(RookTable.data()), Magics, true);
    init_magics(BISHOP, const_cast<MagicMask*>(BishopTable.data()), Magics, true);
}

const Magic& magic(Square s, PieceType pt) {
    assert((pt == BISHOP || pt == ROOK) && is_ok(s));
    return Magics[s][pt - BISHOP];
}

Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    return magic(s, BISHOP).attacks_bb(occupied);
}

Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    return magic(s, ROOK).attacks_bb(occupied);
}

std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied) {
    return {bishop_attacks_bb(s, occupied), rook_attacks_bb(s, occupied)};
}

}  // namespace Stockfish::Attacks

#endif
