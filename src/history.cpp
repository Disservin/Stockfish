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

#include "history.h"

#include <algorithm>

#include "types.h"
#include "position.h"
#include "search.h"

namespace Stockfish {

int stat_bonus(Depth d) { return std::min(253 * d - 356, 1117); }

int stat_malus(Depth d) { return std::min(517 * d - 308, 1206); }

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
                      Depth           depth) {

    Color                  us             = pos.side_to_move();
    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece                  moved_piece    = pos.moved_piece(bestMove);
    PieceType              captured;

    int quietMoveBonus = stat_bonus(depth + 1);
    int quietMoveMalus = stat_malus(depth);

    if (!pos.capture_stage(bestMove))
    {
        int bestMoveBonus = bestValue > beta + 167 ? quietMoveBonus      // larger bonus
                                                   : stat_bonus(depth);  // smaller bonus

        // Increase stats for the best move in case it was a quiet move
        update_quiet_stats(pos, ss, workerThread, bestMove, bestMoveBonus);

        int pIndex = pawn_structure_index(pos);
        workerThread.pawnHistory[pIndex][moved_piece][bestMove.to_sq()] << quietMoveBonus;

        // Decrease stats for all non-best quiet moves
        for (int i = 0; i < quietCount; ++i)
        {
            workerThread
                .pawnHistory[pIndex][pos.moved_piece(quietsSearched[i])][quietsSearched[i].to_sq()]
              << -quietMoveMalus;

            workerThread.mainHistory[us][quietsSearched[i].from_to()] << -quietMoveMalus;
            update_continuation_histories(ss, pos.moved_piece(quietsSearched[i]),
                                          quietsSearched[i].to_sq(), -quietMoveMalus);
        }
    }
    else
    {
        // Increase stats for the best move in case it was a capture move
        captured = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[moved_piece][bestMove.to_sq()][captured] << quietMoveBonus;
    }

    // Extra penalty for a quiet early move that was not a TT move or
    // main killer move in previous ply when it gets refuted.
    if (prevSq != SQ_NONE
        && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit
            || ((ss - 1)->currentMove == (ss - 1)->killers[0]))
        && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -quietMoveMalus);

    // Decrease stats for all non-best capture moves
    for (int i = 0; i < captureCount; ++i)
    {
        moved_piece = pos.moved_piece(capturesSearched[i]);
        captured    = type_of(pos.piece_on(capturesSearched[i].to_sq()));
        captureHistory[moved_piece][capturesSearched[i].to_sq()][captured] << -quietMoveMalus;
    }
}

void update_continuation_histories(Search::Stack* ss, Piece pc, Square to, int bonus) {

    for (int i : {1, 2, 3, 4, 6})
    {
        // Only update the first 2 continuation histories if we are in check
        if (ss->inCheck && i > 2)
            break;
        if (((ss - i)->currentMove).is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << bonus / (1 + 3 * (i == 3));
    }
}

void update_quiet_stats(
  const Position& pos, Search::Stack* ss, Search::Worker& workerThread, Move move, int bonus) {

    // Update killers
    if (ss->killers[0] != move)
    {
        ss->killers[1] = ss->killers[0];
        ss->killers[0] = move;
    }

    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.from_to()] << bonus;
    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus);

    // Update countermove history
    if (((ss - 1)->currentMove).is_ok())
    {
        Square prevSq                                           = ((ss - 1)->currentMove).to_sq();
        workerThread.counterMoves[pos.piece_on(prevSq)][prevSq] = move;
    }
}
}
