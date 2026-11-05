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
#include "king_pins.h"

#include <cassert>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../position.h"

namespace Stockfish::Eval::NNUE::Features {

sf_always_inline IndexType KingPins::make_index(Color perspective, Square ksq, u16 raw) {
    const u8    orientation = OrientTBL[ksq] ^ u8(56 * perspective);
    const Color king        = KingPinList::king(raw);
    const Piece blocker     = KingPinList::blocker(raw);

    assert(type_of(blocker) >= PAWN && type_of(blocker) <= QUEEN);
    assert(KingPinList::slider(raw) >= BISHOP && KingPinList::slider(raw) <= QUEEN);

    const IndexType kingId    = king != perspective;
    const IndexType sliderId  = KingPinList::slider(raw) - BISHOP;
    const IndexType blockerId = (type_of(blocker) - PAWN) * 2 + (color_of(blocker) != perspective);
    const IndexType square    = u8(KingPinList::square(raw)) ^ orientation;

    return ((kingId * NumSliders + sliderId) * NumBlockers + blockerId) * SQUARE_NB + square;
}

void KingPins::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    const Square ksq = pos.square<KING>(perspective);

    for (Color king : {WHITE, BLACK})
    {
        const Square kingSq  = pos.square<KING>(king);
        Bitboard     snipers = ((Attacks::attacks_bb(ROOK, kingSq) & pos.pieces(QUEEN, ROOK))
                                | (Attacks::attacks_bb(BISHOP, kingSq) & pos.pieces(QUEEN, BISHOP)))
                             & pos.pieces(~king);
        const Bitboard occupancy = pos.pieces() ^ snipers;

        while (snipers)
        {
            const Square   sniperSq = pop_lsb(snipers);
            const Bitboard b        = Attacks::between_bb(kingSq, sniperSq) & occupancy;

            if (b && !more_than_one(b) && !(b & pos.pieces(KING)))
                active.push_back(
                  make_index(perspective, ksq,
                             KingPinList::encode(king, type_of(pos.piece_on(sniperSq)),
                                                 pos.piece_on(lsb(b)), lsb(b))));
        }
    }
}

void KingPins::append_changed_indices(
  Color perspective, Square ksq, const DiffType& diff, IndexList& removed, IndexList& added) {
    const KingPinList& before = *diff.before;
    const KingPinList& after  = *diff.after;

    // Both lists are tiny and a raw value can appear twice (two sliders of the same
    // type behind one blocker), so match them up as multisets.
    u32 unmatched = (u32(1) << after.count) - 1;

    for (int i = 0; i < before.count; ++i)
    {
        bool found = false;
        for (u32 m = unmatched; m; m &= m - 1)
        {
            const int j = int(lsb(Bitboard(m)));
            if (after.raw[j] == before.raw[i])
            {
                unmatched &= ~(u32(1) << j);
                found = true;
                break;
            }
        }
        if (!found)
            removed.push_back(make_index(perspective, ksq, before.raw[i]));
    }

    for (u32 m = unmatched; m; m &= m - 1)
        added.push_back(make_index(perspective, ksq, after.raw[int(lsb(Bitboard(m)))]));
}

}  // namespace Stockfish::Eval::NNUE::Features
