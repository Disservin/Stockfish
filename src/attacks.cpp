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

#include <array>

#include "misc.h"

namespace Stockfish::Attacks {

namespace {

Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard RayPassBB[SQUARE_NB][SQUARE_NB];

#ifdef USE_DUAL_HYPERBOLA_QUINT
alignas(64) DualMagic DualMagics[SQUARE_NB];
#else
alignas(64) Magic Magics[SQUARE_NB][2];
#endif

}

#ifdef USE_PEXT
using MagicMask = uint16_t;
#else
using MagicMask = Bitboard;
#endif

[[maybe_unused]] static Bitboard line_mask(Square sq, Direction d1, Direction d2) {
    Bitboard mask = 0, dest;
    for (Direction d : {d1, d2})
    {
        Square s = sq;
        while ((dest = safe_destination(s, d)))
        {
            mask |= dest;
            s += d;
        }
    }
    return mask;
}

#ifdef USE_HYPERBOLA_QUINT
#include "attacks_hyperbola_impl.h"
#elif defined(USE_DUAL_HYPERBOLA_QUINT)
#include "attacks_dual_hyperbola_impl.h"
#else
#include "attacks_magic_impl.h"
#endif

void init() {

#ifdef USE_HYPERBOLA_QUINT
    init_magics(Magics);
#elif defined(USE_DUAL_HYPERBOLA_QUINT)
    init_dual_magics(DualMagics);
#else
    init_magics(ROOK, const_cast<MagicMask*>(RookTable.data()), Magics, true);
    init_magics(BISHOP, const_cast<MagicMask*>(BishopTable.data()), Magics, true);
#endif

    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
    {
        for (PieceType pt : {BISHOP, ROOK})
            for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2)
            {
                if (PseudoAttacks[pt][s1] & s2)
                {
                    LineBB[s1][s2] = (attacks_bb(pt, s1, 0) & attacks_bb(pt, s2, 0)) | s1 | s2;
                    BetweenBB[s1][s2] =
                      (attacks_bb(pt, s1, square_bb(s2)) & attacks_bb(pt, s2, square_bb(s1)));
                    RayPassBB[s1][s2] =
                      attacks_bb(pt, s1, 0) & (attacks_bb(pt, s2, square_bb(s1)) | s2);
                }
                BetweenBB[s1][s2] |= s2;
            }
    }
}

#ifdef USE_DUAL_HYPERBOLA_QUINT
const DualMagic& dual_magic(Square s) { return DualMagics[s]; }
#else
const Magic& magic(Square s, PieceType pt) {
    assert((pt == BISHOP || pt == ROOK) && is_ok(s));
    return Magics[s][pt - BISHOP];
}
#endif

Bitboard line_bb(Square s1, Square s2) {
    assert(is_ok(s1) && is_ok(s2));
    return LineBB[s1][s2];
}

Bitboard between_bb(Square s1, Square s2) {
    assert(is_ok(s1) && is_ok(s2));
    return BetweenBB[s1][s2];
}

Bitboard ray_pass_bb(Square s1, Square s2) {
    assert(is_ok(s1) && is_ok(s2));
    return RayPassBB[s1][s2];
}

}  // namespace Stockfish::Attacks
