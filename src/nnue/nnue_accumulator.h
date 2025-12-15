/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

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

// Class for difference calculation of NNUE evaluation function

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

#include "nnue_accumulator_new.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {


class AccumulatorStack {
   public:
    template<IndexType Dimensions>
    [[nodiscard]] const auto& get() const noexcept {
        if constexpr (Dimensions == TransformedFeatureDimensionsBig)
        {
            return big;
        }
        else
        {
            return small;
        }
    }

    void reset() noexcept {
        big.reset();
        small.reset();
    }

    std::pair<DirtyPiece&, DirtyThreats&> get_diff_type() noexcept {
        auto& dts = big.threat.reset().diff;
        auto& dp  = big.psqt.reset().diff;

        new (&dts) DirtyThreats;

        return {dp, dts};
    }

    void propagate_changes() noexcept { small.psqt.reset().diff = big.psqt.latest().diff; }

    void pop() noexcept {
        big.pop();
        small.pop();
    }

    template<IndexType Dimensions>
    void evaluate(const Position&                       pos,
                  const FeatureTransformer<Dimensions>& featureTransformer,
                  AccumulatorCaches::Cache<Dimensions>& cache) noexcept {
        if constexpr (Dimensions == TransformedFeatureDimensionsBig)
        {
            big.evaluate(pos, featureTransformer, cache);
        }
        else
        {
            small.evaluate(pos, featureTransformer, cache);
        }
    }

   private:
    BigNetworkAccumulator   big;
    SmallNetworkAccumulator small;
};

}  // namespace Stockfish::Eval::NNUE

#endif  // NNUE_ACCUMULATOR_H_INCLUDED
