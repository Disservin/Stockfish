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

#ifndef MOVEPICK_H_INCLUDED
#define MOVEPICK_H_INCLUDED

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>  // IWYU pragma: keep

#include "movegen.h"
#include "types.h"
#include "position.h"
#include "history.h"

namespace Stockfish {

// MovePicker class is used to pick one pseudo-legal move at a time from the
// current position. The most important method is next_move(), which returns a
// new pseudo-legal move each time it is called, until there are no moves left,
// when Move::none() is returned. In order to improve the efficiency of the
// alpha-beta algorithm, MovePicker attempts to return the moves which are most
// likely to get a cut-off first.
class MovePicker {

    enum PickType {
        Next,
        Best
    };

   public:
    MovePicker(const MovePicker&)            = delete;
    MovePicker& operator=(const MovePicker&) = delete;
    MovePicker(const Position&,
               Move,
               Depth,
               const ButterflyHistory*,
               const CapturePieceToHistory*,
               const PieceToHistory**,
               const PawnHistory*,
               Move,
               const Move*);
    MovePicker(const Position&,
               Move,
               Depth,
               const ButterflyHistory*,
               const CapturePieceToHistory*,
               const PieceToHistory**,
               const PawnHistory*);
    MovePicker(const Position&, Move, int, const CapturePieceToHistory*);
    Move next_move(bool skipQuiets = false);

   private:
    template<PickType T, typename Pred>
    Move select(Pred);
    template<GenType>
    void     score();
    ExtMove* begin() { return cur; }
    ExtMove* end() { return endMoves; }

    const Position&              pos;
    const ButterflyHistory*      mainHistory;
    const CapturePieceToHistory* captureHistory;
    const PieceToHistory**       continuationHistory;
    const PawnHistory*           pawnHistory;
    Move                         ttMove;
    ExtMove refutations[3], *cur, *endMoves, *endBadCaptures, *beginBadQuiets, *endBadQuiets;
    int     stage;
    int     threshold;
    Depth   depth;
    ExtMove moves[MAX_MOVES];
};

}  // namespace Stockfish

#endif  // #ifndef MOVEPICK_H_INCLUDED
