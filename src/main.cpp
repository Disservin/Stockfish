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

#include <iostream>
#include <unordered_map>

#include "misc.h"
#include "position.h"
#include "syzygy/tbprobe.h"
#include "types.h"

using namespace Stockfish;

int main() {
    Bitboards::init();
    Position::init();

    Tablebases::init("./syzygy");

    StateListPtr states(new std::deque<StateInfo>(1));

    Position pos;
    pos.set("8/8/3K4/1r6/8/8/4k3/2R5 b - - 0 18", false, &states->back());

    Tablebases::ProbeState score = Tablebases::ProbeState::OK;
    Tablebases::probe_dtz(pos, &score);


    return 0;
}
