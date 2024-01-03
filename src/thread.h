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

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "position.h"
#include "thread_win32_osx.h"
#include "types.h"
#include "timeman.h"
#include "search.h"

namespace Stockfish {

class ThreadPool;

// Thread class keeps together all the thread-related stuff.
class Thread {
   public:
    OptionsMap&         options;
    TranspositionTable& tt;
    ThreadPool&         threads;

   private:
    std::mutex              mutex;
    std::condition_variable cv;
    size_t                  idx;
    bool                    exit = false, searching = true;  // Set before starting std::thread
    NativeThread            stdThread;

   public:
    Thread(OptionsMap&, TranspositionTable&, ThreadPool&, size_t);
    virtual ~Thread();
    virtual void search();
    void         clear();
    void         idle_loop();
    void         start_searching();
    void         wait_for_search_finished();
    size_t       id() const { return idx; }


    Search::LimitsType limits;

    size_t                pvIdx, pvLast;
    std::atomic<uint64_t> nodes, tbHits, bestMoveChanges;
    int                   selDepth, nmpMinPly;
    Value                 bestValue, optimism[COLOR_NB];

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
};


// MainThread is a derived class specific for main thread
struct MainThread: public Thread {

    using Thread::Thread;

    void search() override;
    void check_time();

    TimeManagement tm;

    double           previousTimeReduction;
    Value            bestPreviousScore;
    Value            bestPreviousAverageScore;
    Value            iterValue[4];
    int              callsCnt;
    bool             stopOnPonderhit;
    std::atomic_bool ponder;
};

// ThreadPool struct handles all the threads-related stuff like init, starting,
// parking and, most importantly, launching a thread. All the access to threads
// is done through this class.
class ThreadPool {

   public:
    ThreadPool(OptionsMap& o, TranspositionTable& t) :
        options(o),
        tt(t) {}

    void start_thinking(Position&, StateListPtr&, Search::LimitsType, bool = false);
    void clear();
    void set(size_t);

    MainThread* main() const { return static_cast<MainThread*>(threads.front()); }
    uint64_t    nodes_searched() const { return accumulate(&Thread::nodes); }
    uint64_t    tb_hits() const { return accumulate(&Thread::tbHits); }
    Thread*     get_best_thread() const;
    void        start_searching();
    void        wait_for_search_finished() const;

    std::atomic_bool stop, increaseDepth;

    auto cbegin() const noexcept { return threads.cbegin(); }
    auto begin() noexcept { return threads.begin(); }
    auto end() noexcept { return threads.end(); }
    auto cend() const noexcept { return threads.cend(); }
    auto size() const noexcept { return threads.size(); }
    auto empty() const noexcept { return threads.empty(); }

   private:
    OptionsMap&         options;
    TranspositionTable& tt;

    StateListPtr         setupStates;
    std::vector<Thread*> threads;

    uint64_t accumulate(std::atomic<uint64_t> Thread::*member) const {

        uint64_t sum = 0;
        for (Thread* th : threads)
            sum += (th->*member).load(std::memory_order_relaxed);
        return sum;
    }
};

}  // namespace Stockfish

#endif  // #ifndef THREAD_H_INCLUDED
