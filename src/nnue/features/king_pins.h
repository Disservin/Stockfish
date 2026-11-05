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
#ifndef NNUE_FEATURES_KING_PINS_INCLUDED
#define NNUE_FEATURES_KING_PINS_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

// Feature KingPins: an enemy slider aimed at a king with exactly one non-king piece between
// them, when other enemy sliders on the king's lines are ignored. Indexed by whose king it is,
// the slider type, the piece in between and its square. These are the blockers computed by
// Position::update_slider_blockers(), of either color, so they cover both pins and discovered
// attacks.
class KingPins {
    static constexpr IndexType NumKings    = 2;
    static constexpr IndexType NumSliders  = 3;   // bishop, rook, queen
    static constexpr IndexType NumBlockers = 10;  // pawn..queen of either color

   public:
    // Hash value embedded in the evaluation file
    static constexpr u32 HashValue = 0x7b1e4c93u;

    // Number of feature dimensions
    static constexpr IndexType Dimensions =
      NumKings * NumSliders * NumBlockers * static_cast<IndexType>(SQUARE_NB);

    // Maximum number of simultaneously active features.
    static constexpr IndexType MaxActiveDimensions = KingPinList::Capacity;
    using IndexList                                = ValueList<u16, MaxActiveDimensions>;
    using DiffType                                 = DirtyKingPins;

    static IndexType make_index(Color perspective, Square ksq, u16 raw);

    // Get a list of indices for active features, computed from the board
    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);

    // Whether any feature could have changed. Usually false, so inlined to keep it cheap.
    static bool may_change(const DiffType& diff) {
        return (diff.before->count | diff.after->count) != 0;
    }

    // Get a list of indices for recently changed features
    static void append_changed_indices(
      Color perspective, Square ksq, const DiffType& diff, IndexList& removed, IndexList& added);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_KING_PINS_INCLUDED
