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

#include "normalization.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace Stockfish::Normalization {

namespace {

struct WinRateParams {
    double a;
    double b;
};

WinRateParams win_rate_params(int material) {

    // The fitted model only uses data for material counts in [10, 78], and is anchored at count 58.
    double m = std::clamp(material, 10, 78) / 58.0;

    // Return a = p_a(material) and b = p_b(material), see github.com/official-stockfish/WDL_model
    constexpr double as[] = {-185.71965483, 504.85014385, -438.58295743, 474.04604627};
    constexpr double bs[] = {89.23542728, -137.02141296, 73.28669021, 47.53376190};

    double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
    double b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

    return {a, b};
}

// The win rate model is 1 / (1 + exp((a - eval) / b)), where a = p_a(material) and b = p_b(material).
// It fits the LTC fishtest statistics rather accurately.
int win_rate_model(Value v, int material) {

    auto [a, b] = win_rate_params(material);

    // Return the win rate in per mille units, rounded to the nearest integer.
    return int(0.5 + 1000 / (1 + std::exp((a - double(v)) / b)));
}
}

std::string wdl(Value v, int material) {
    std::stringstream ss;

    int wdl_w = win_rate_model(v, material);
    int wdl_l = win_rate_model(-v, material);
    int wdl_d = 1000 - wdl_w - wdl_l;
    ss << " wdl " << wdl_w << " " << wdl_d << " " << wdl_l;

    return ss.str();
}

// Turns a Value to an integer centipawn number,
// without treatment of mate and similar special scores.
int to_cp(Value v, int material) {
    // In general, the score can be defined via the the WDL as
    // (log(1/L - 1) - log(1/W - 1)) / ((log(1/L - 1) + log(1/W - 1))
    // Based on our win_rate_model, this simply yields v / a.

    auto [a, b] = win_rate_params(material);

    return std::round(100 * int(v) / a);
}

}