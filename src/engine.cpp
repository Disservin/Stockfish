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

#include "engine.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

#include "benchmark.h"
#include "evaluate.h"
#include "movegen.h"
#include "nnue/network.h"
#include "nnue/nnue_common.h"
#include "perft.h"
#include "position.h"
#include "search.h"
#include "syzygy/tbprobe.h"
#include "types.h"
#include "ucioption.h"

namespace Stockfish {

namespace NN = Eval::NNUE;

constexpr auto StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

Engine::Engine(std::string path) :
    binaryDirectory(CommandLine::get_binary_directory(path)),
    states(new std::deque<StateInfo>(1)),
    networks(NN::Networks(
      NN::NetworkBig({EvalFileDefaultNameBig, "None", ""}, NN::EmbeddedNNUEType::BIG),
      NN::NetworkSmall({EvalFileDefaultNameSmall, "None", ""}, NN::EmbeddedNNUEType::SMALL))) {
    Tune::init(options);
    pos.set(StartFEN, false, &states->back());
}

void Engine::go(const Search::LimitsType& limits) {
    verify_networks();

    if (limits.perft)
    {
        perft(pos.fen(), limits.perft, options["UCI_Chess960"]);
        return;
    }

    threads.start_thinking(options, pos, states, limits);
}
void Engine::stop() { threads.stop = true; }

void Engine::search_clear() {
    wait_for_search_finished();

    tt.clear(options["Threads"]);
    threads.clear();

    // @TODO wont work multiple instances
    Tablebases::init(options["SyzygyPath"]);  // Free mapped files
}

void Engine::wait_for_search_finished() { threads.main_thread()->wait_for_search_finished(); }

void Engine::set_position(const std::string& fen, const std::vector<std::string>& moves) {
    // Drop the old state and create a new one
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(fen, options["UCI_Chess960"], &states->back());

    for (const auto& move : moves)
    {
        auto m = UCI::to_move(pos, move);

        if (m == Move::none())
            break;

        pos.do_move(m, states->back());
    }
}

// modifiers

void Engine::resize_threads() { threads.set({options, threads, tt, networks}); }

void Engine::set_tt_size(size_t mb) {
    wait_for_search_finished();
    tt.resize(mb, options["Threads"]);
}

void Engine::set_ponderhit(bool b) { threads.main_manager()->ponder = b; }

// network related

void Engine::verify_networks() {
    networks.big.verify(options["EvalFile"]);
    networks.small.verify(options["EvalFileSmall"]);
}

void Engine::load_networks() {
    networks.big.load(binaryDirectory, options["EvalFile"]);
    networks.small.load(binaryDirectory, options["EvalFileSmall"]);
}

void Engine::load_big_network(const std::string& file) { networks.big.load(binaryDirectory, file); }

void Engine::load_small_network(const std::string& file) {
    networks.small.load(binaryDirectory, file);
}

void Engine::save_network(const std::string& file) { networks.big.save(file); }

// utility functions

OptionsMap& Engine::get_options() { return options; }

uint64_t Engine::nodes_searched() const { return threads.nodes_searched(); }

void Engine::trace_eval() {
    StateListPtr trace_states(new std::deque<StateInfo>(1));
    Position     p;
    p.set(pos.fen(), options["UCI_Chess960"], &trace_states->back());

    verify_networks();

    sync_cout << "\n" << Eval::trace(p, networks) << sync_endl;
}

}