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

// Class for difference calculation of NNUE evaluation function

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include <cstdint>

#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE {

using BiasType       = std::int16_t;
using PSQTWeightType = std::int32_t;
using IndexType      = std::uint32_t;

// Class that holds the result of affine transformation of input features
template<IndexType Size>
struct alignas(CacheLineSize) Accumulator {
    std::int16_t accumulation[2][Size];
    std::int32_t psqtAccumulation[2][PSQTBuckets];
    bool         computed[2];
    bool         computedPSQT[2];
};


// AccumulatorCaches struct provides per-thread accumulator caches, where each
// cache contains multiple entries for each of the possible king squares.
// When the accumulator needs to be refreshed, the cached entry is used to more
// efficiently update the accumulator, instead of rebuilding it from scratch.
// This idea, was first described by Luecx (author of Koivisto) and
// is commonly referred to as "Finny Tables".
struct AccumulatorCaches {

    template<IndexType Size>
    struct alignas(CacheLineSize) Cache {

        struct alignas(CacheLineSize) Entry {
            BiasType       accumulation[2][Size];
            PSQTWeightType psqtAccumulation[2][PSQTBuckets];
            Bitboard       byColorBB[COLOR_NB][COLOR_NB];
            Bitboard       byTypeBB[COLOR_NB][PIECE_TYPE_NB];

            // To initialize a refresh entry, we set all its bitboards empty,
            // so we put the biases in the accumulation, without any weights on top
            void clear(const BiasType* biases) {

                std::memset(byColorBB, 0, 2 * 2 * sizeof(Bitboard));
                std::memset(byTypeBB, 0, 2 * 8 * sizeof(Bitboard));

                std::memcpy(accumulation[WHITE], biases, Size * sizeof(BiasType));
                std::memcpy(accumulation[BLACK], biases, Size * sizeof(BiasType));

                std::memset(psqtAccumulation, 0, sizeof(psqtAccumulation));
            }
        };

        template<typename Network>
        void clear(const Network& network) {
            for (auto& entry : entries)
                entry.clear(network.featureTransformer->biases);
        }

        void clear(const BiasType* biases) {
            for (auto& entry : entries)
                entry.clear(biases);
        }

        Entry& operator[](Square sq) { return entries[sq]; }

        std::array<Entry, SQUARE_NB> entries;
    };

    template<typename Networks>
    void clear(const Networks& networks) {
        big.clear(networks.big);
    }

    // When adding a new cache for a network, i.e. the smallnet
    // the appropriate condition must be added to FeatureTransformer::update_accumulator_refresh.
    Cache<TransformedFeatureDimensionsBig> big;
};

}  // namespace Stockfish::Eval::NNUE

#endif  // NNUE_ACCUMULATOR_H_INCLUDED
