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

#ifndef NEW_UCI_H_INCLUDED
#define NEW_UCI_H_INCLUDED

#include <string>
#include <functional>
#include <map>
#include <iostream>
#include <deque>

#include "types.h"
#include "thread.h"
#include "tt.h"

#include "options_map.h"

namespace Stockfish {


class StateInfo;
class Position;
// class ThreadPool;
using TimePoint    = std::chrono::milliseconds::rep;  // A value in milliseconds
using StateListPtr = std::unique_ptr<std::deque<StateInfo>>;

class NewUci {
   public:
    NewUci();
    ~NewUci();
    void               loop(int argc, char* argv[]);
    static int         to_cp(Value v);
    static std::string value(Value v);
    static std::string square(Square s);
    static std::string move(Move m, bool chess960);
    static std::string pv(const Position& pos, Depth depth, TimePoint elapsed);
    std::string        wdl(Value v, int ply);
    static Move        to_move(const Position& pos, std::string& str);

    OptionsMap         options;
    TranspositionTable tt;
    ThreadPool         threads;

   private:
    void go(Position& pos, std::istringstream& is, StateListPtr& states);
    void bench(Position& pos, std::istream& args, StateListPtr& states);
    void position(Position& pos, std::istringstream& is, StateListPtr& states);
    void trace_eval(Position& pos);

    void search_clear();
};

}  // namespace Stockfish


#endif