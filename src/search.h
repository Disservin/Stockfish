/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2023 The Stockfish developers (see AUTHORS file)

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

#ifndef SEARCH_H_INCLUDED
#define SEARCH_H_INCLUDED

#include <cstdint>
#include <vector>
#include <atomic>

#include "misc.h"
#include "movepick.h"
#include "types.h"

namespace Stockfish {

// Different node types, used as a template parameter
enum NodeType {
    NonPV,
    PV,
    Root
};

class Position;
class TranspositionTable;
class ThreadPool;


namespace Search {

struct ExternalShared {
    ExternalShared(const OptionsMap& o, ThreadPool& tp, TranspositionTable& t) :
        options(o),
        threads(tp),
        tt(t) {}

    const OptionsMap&   options;
    ThreadPool&         threads;
    TranspositionTable& tt;
};


// Stack struct keeps track of the information we need to remember from nodes
// shallower and deeper in the tree during the search. Each search thread has
// its own array of Stack objects, indexed by the current ply.
struct Stack {
    Move*           pv;
    PieceToHistory* continuationHistory;
    int             ply;
    Move            currentMove;
    Move            excludedMove;
    Move            killers[2];
    Value           staticEval;
    int             statScore;
    int             moveCount;
    bool            inCheck;
    bool            ttPv;
    bool            ttHit;
    int             doubleExtensions;
    int             cutoffCnt;
};


// RootMove struct is used for moves at the root of the tree. For each root move
// we store a score and a PV (really a refutation in the case of moves which
// fail low). Score is normally set at -VALUE_INFINITE for all non-pv moves.
struct RootMove {

    explicit RootMove(Move m) :
        pv(1, m) {}
    bool extract_ponder_from_tt(TranspositionTable& tt, Position& pos);
    bool operator==(const Move& m) const { return pv[0] == m; }
    // Sort in descending order
    bool operator<(const RootMove& m) const {
        return m.score != score ? m.score < score : m.previousScore < previousScore;
    }

    Value             score           = -VALUE_INFINITE;
    Value             previousScore   = -VALUE_INFINITE;
    Value             averageScore    = -VALUE_INFINITE;
    Value             uciScore        = -VALUE_INFINITE;
    bool              scoreLowerbound = false;
    bool              scoreUpperbound = false;
    int               selDepth        = 0;
    int               tbRank          = 0;
    Value             tbScore;
    std::vector<Move> pv;
};

using RootMoves = std::vector<RootMove>;


// LimitsType struct stores information sent by GUI about available time to
// search the current move, maximum depth/time, or if we are in analysis mode.

struct LimitsType {

    // Init explicitly due to broken value-initialization of non POD in MSVC
    LimitsType() {
        time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime = TimePoint(0);
        movestogo = depth = mate = perft = infinite = 0;
        nodes                                       = 0;
    }

    bool use_time_management() const { return time[WHITE] || time[BLACK]; }

    std::vector<Move> searchmoves;
    TimePoint         time[COLOR_NB], inc[COLOR_NB], npmsec, movetime, startTime;
    int               movestogo, depth, mate, perft, infinite;
    int64_t           nodes;
};


void init(int);

}  // namespace Search


class SearchWorker {
   public:
    SearchWorker(Search::ExternalShared& es) :
        // Unpack the ExternalShared struct into member variables
        options(es.options),
        threads(es.threads),
        tt(es.tt) {}

    Search::LimitsType limits;

    size_t                pvIdx, pvLast;
    std::atomic<uint64_t> nodes, tbHits, bestMoveChanges;
    int                   selDepth, nmpMinPly;
    Value                 iterBestValue, optimism[COLOR_NB];

    Position              rootPos;
    StateInfo             rootState;
    Search::RootMoves     rootMoves;
    Depth                 rootDepth, completedDepth;
    Value                 rootDelta;
    Value                 rootSimpleEval;
    CounterMoveHistory    counterMoves;
    ButterflyHistory      mainHistory;
    CapturePieceToHistory captureHistory;
    ContinuationHistory   continuationHistory[2][2];
    PawnHistory           pawnHistory;
    CorrectionHistory     correctionHistory;

   protected:
    template<NodeType nodeType>
    Value
    search(Position& pos, Search::Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode);

    const OptionsMap&   options;
    ThreadPool&         threads;
    TranspositionTable& tt;

   private:
    template<NodeType nodeType>
    Value qsearch(Position& pos, Search::Stack* ss, Value alpha, Value beta, Depth depth = 0);
};

}  // namespace Stockfish

#endif  // #ifndef SEARCH_H_INCLUDED
