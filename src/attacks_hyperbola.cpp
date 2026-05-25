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

#include "attacks_hyperbola.h"

#ifdef USE_HYPERBOLA_QUINT

    #ifdef __aarch64__
        #include <arm_acle.h>
    #endif

namespace Stockfish::Attacks {
namespace {

alignas(64) Magic Magics[SQUARE_NB][2];

Bitboard reverse_bb(Bitboard bb) {
    #ifdef __aarch64__
    return __rbitll(bb);
    #else  // loongarch
    Bitboard out;
    asm("bitrev.d %0, %1" : "=r"(out) : "r"(bb));
    return out;
    #endif
}

Bitboard attacks_bb(const Magic& m, Bitboard occupied) {
    const auto hyperbola = [&](Bitboard mask) {
        Bitboard o   = occupied & mask;
        Bitboard fwd = o - m.r;
        Bitboard rev = reverse_bb(o) - m.rr;
        return (fwd ^ reverse_bb(rev)) & mask;
    };

    return hyperbola(m.mask1) | hyperbola(m.mask2);
}

}  // namespace

void init_impl() {
    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        Magic& rook = Magics[s][ROOK - BISHOP];
        rook.mask1  = line_mask(s, NORTH, SOUTH);
        rook.mask2  = line_mask(s, EAST, WEST);

        Magic& bishop = Magics[s][BISHOP - BISHOP];
        bishop.mask1  = line_mask(s, NORTH_EAST, SOUTH_WEST);
        bishop.mask2  = line_mask(s, NORTH_WEST, SOUTH_EAST);

        rook.r = bishop.r = square_bb(s) * 2;
        rook.rr = bishop.rr = square_bb(Square(63 - int(s))) * 2;
    }
}

const Magic& magic(Square s, PieceType pt) {
    assert((pt == BISHOP || pt == ROOK) && is_ok(s));
    return Magics[s][pt - BISHOP];
}

Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    return attacks_bb(magic(s, BISHOP), occupied);
}

Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    return attacks_bb(magic(s, ROOK), occupied);
}

std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied) {
    return {bishop_attacks_bb(s, occupied), rook_attacks_bb(s, occupied)};
}

}  // namespace Stockfish::Attacks

#endif
