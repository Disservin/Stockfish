/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2022 The Stockfish developers (see AUTHORS file)

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

#include <vector>

#include "misc.h"
#include "movepick.h"
#include "types.h"

namespace Stockfish {

class Position;

namespace Search {

// Different node types, used as a template parameter
enum NodeType { NonPV, PV, Root };

/// Stack struct keeps track of the information we need to remember from nodes
/// shallower and deeper in the tree during the search. Each search thread has
/// its own array of Stack objects, indexed by the current ply.

struct Stack {
  Move *pv;
  PieceToHistory *continuationHistory;
  int ply;
  Move currentMove;
  Move excludedMove;
  Move killers[2];
  Value staticEval;
  int statScore;
  int moveCount;
  bool inCheck;
  bool ttPv;
  bool ttHit;
  int doubleExtensions;
  int cutoffCnt;
};

/// RootMove struct is used for moves at the root of the tree. For each root
/// move we store a score and a PV (really a refutation in the case of moves
/// which fail low). Score is normally set at -VALUE_INFINITE for all non-pv
/// moves.

struct RootMove {
  explicit RootMove(Move m) : pv(1, m) {}
  bool extract_ponder_from_tt(Position &pos);
  bool operator==(const Move &m) const { return pv[0] == m; }
  bool operator<(const RootMove &m) const {  // Sort in descending order
    return m.score != score ? m.score < score : m.previousScore < previousScore;
  }

  Value score = -VALUE_INFINITE;
  Value previousScore = -VALUE_INFINITE;
  Value averageScore = -VALUE_INFINITE;
  int selDepth = 0;
  int tbRank = 0;
  Value tbScore;
  std::vector<Move> pv;
};

typedef std::vector<RootMove> RootMoves;

/// LimitsType struct stores information sent by GUI about available time to
/// search the current move, maximum depth/time, or if we are in analysis mode.

struct LimitsType {
  LimitsType() {  // Init explicitly due to broken value-initialization of non
                  // POD in MSVC
    time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = npmsec = movetime =
        TimePoint(0);
    movestogo = depth = mate = perft = infinite = 0;
    nodes = 0;
  }

  bool use_time_management() const { return time[WHITE] || time[BLACK]; }

  std::vector<Move> searchmoves;
  TimePoint time[COLOR_NB], inc[COLOR_NB], npmsec, movetime, startTime;
  int movestogo, depth, mate, perft, infinite;
  int64_t nodes;
};

extern LimitsType Limits;

template <NodeType nodeType>
Value search(Position &pos, Stack *ss, Value alpha, Value beta, Depth depth,
             bool cutNode);

template <NodeType nodeType>
Value qsearch(Position &pos, Stack *ss, Value alpha, Value beta,
              Depth depth = 0);

void update_all_stats(const Position &pos, Stack *ss, Move bestMove,
                      Value bestValue, Value beta, Square prevSq,
                      Move *quietsSearched, int quietCount,
                      Move *capturesSearched, int captureCount, Depth depth);

void update_continuation_histories(Stack *ss, Piece pc, Square to, int bonus);
void update_quiet_stats(const Position &pos, Stack *ss, Move move, int bonus);

Value futility_margin(Depth d, bool improving);
Depth reduction(bool i, Depth d, int mn, Value delta, Value rootDelta);
constexpr int futility_move_count(bool improving, Depth depth);
int stat_bonus(Depth d);
Value value_draw(const Thread *thisThread);

void update_pv(Move *pv, Move move, const Move *childPv);

void init();
void clear();

Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r50c);
struct Skill {
  Skill(int skill_level, int uci_elo);
  bool enabled() const { return level < 20.0; }
  bool time_to_pick(Depth depth) const { return depth == 1 + int(level); }
  Move pick_best(size_t multiPV);

  double level;
  Move best = MOVE_NONE;
};

template <bool Root>
uint64_t perft(Position &pos, Depth depth);

}  // namespace Search

}  // namespace Stockfish

#endif  // #ifndef SEARCH_H_INCLUDED
