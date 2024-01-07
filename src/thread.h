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

#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "position.h"
#include "search.h"
#include "thread_win32_osx.h"
#include "timeman.h"

namespace Stockfish {

class OptionsMap;
using Value = int;

// Thread class keeps together all the thread-related stuff.
class Thread {
   public:
    Thread(Search::ExternalShared&, std::unique_ptr<Search::ISearchManager>, size_t);
    virtual ~Thread();

    void   idle_loop();
    void   start_searching();
    void   wait_for_search_finished();
    size_t id() const { return idx; }

    std::unique_ptr<Search::Worker> worker;

   private:
    std::mutex              mutex;
    std::condition_variable cv;
    size_t                  idx;
    bool                    exit = false, searching = true;  // Set before starting std::thread
    NativeThread            stdThread;
};


// ThreadPool struct handles all the threads-related stuff like init, starting,
// parking and, most importantly, launching a thread. All the access to threads
// is done through this class.
class ThreadPool {

   public:
    ~ThreadPool() {
        // destroy any existing thread(s)
        if (threads.size() > 0)
        {
            main()->wait_for_search_finished();

            while (threads.size() > 0)
                delete threads.back(), threads.pop_back();
        }
    }

    void
    start_thinking(const OptionsMap&, Position&, StateListPtr&, Search::LimitsType, bool = false);
    void clear();
    void set(Search::ExternalShared&&);

    Search::SearchManager* main_manager() const {
        return dynamic_cast<Search::SearchManager*>(threads.front()->worker.get()->manager.get());
    };
    Thread*  main() const { return threads.front(); }
    uint64_t nodes_searched() const { return accumulate(&Search::Worker::nodes); }
    uint64_t tb_hits() const { return accumulate(&Search::Worker::tbHits); }
    Thread*  get_best_thread() const;
    void     start_searching();
    void     wait_for_search_finished() const;

    std::atomic_bool stop, increaseDepth;

    auto cbegin() const noexcept { return threads.cbegin(); }
    auto begin() noexcept { return threads.begin(); }
    auto end() noexcept { return threads.end(); }
    auto cend() const noexcept { return threads.cend(); }
    auto size() const noexcept { return threads.size(); }
    auto empty() const noexcept { return threads.empty(); }

   private:
    StateListPtr         setupStates;
    std::vector<Thread*> threads;

    uint64_t accumulate(std::atomic<uint64_t> Search::Worker::*member) const {

        uint64_t sum = 0;
        for (Thread* th : threads)
            sum += th->worker.get()->*member;
        return sum;
    }
};

}  // namespace Stockfish

#endif  // #ifndef THREAD_H_INCLUDED
