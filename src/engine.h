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

#ifndef STOCKFISH_H_INCLUDED
#define STOCKFISH_H_INCLUDED

#include "misc.h"
#include "nnue/network.h"
#include "position.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "ucioption.h"

namespace Stockfish {

class Engine {
   public:
    using InfoShort = Search::InfoShort;
    using InfoFull  = Search::InfoFull;
    using InfoIter  = Search::InfoIteration;

    Engine(std::string path = "");

    // non blocking call to start searching
    void go(const Search::LimitsType&);
    // non blocking call to stop searching
    void stop();

    // blocking call to wait for search to finish
    void wait_for_search_finished();
    // set a new position
    void set_position(const std::string& fen, const std::vector<std::string>& moves);

    // modifiers

    void resize_threads();
    void set_tt_size(size_t mb);
    void set_ponderhit(bool);
    // clears the search
    void search_clear();

    void set_on_update_short(std::function<void(const InfoShort&)>);
    void set_on_update_full(std::function<void(const InfoFull&)>);
    void set_on_iter(std::function<void(const InfoIter&)>);
    void set_on_bestmove(std::function<void(const std::string&, const std::string&)>);

    // network related

    void verify_networks();
    void load_networks();
    void load_big_network(const std::string& file);
    void load_small_network(const std::string& file);
    void save_network(const std::string& file);

    // utility functions

    void trace_eval();
    // nodes since last search clear
    uint64_t    nodes_searched() const;
    OptionsMap& get_options();

   private:
    const std::string binaryDirectory;

    Position     pos;
    StateListPtr states;

    OptionsMap           options;
    ThreadPool           threads;
    TranspositionTable   tt;
    Eval::NNUE::Networks networks;

    Search::SearchManager::UpdateContext updateContext;
};

}  // namespace Stockfish


#endif  // #ifndef STOCKFISH_H_INCLUDED