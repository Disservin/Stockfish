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

#ifndef NNUE_ACCUMULATOR_NEW_H_INCLUDED
#define NNUE_ACCUMULATOR_NEW_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {

template<IndexType Size>
struct alignas(CacheLineSize) Accumulator;

template<IndexType TransformedFeatureDimensions>
class FeatureTransformer;

// Class that holds the result of affine transformation of input features
template<IndexType Size>
struct alignas(CacheLineSize) Accumulator {
    std::array<std::array<std::int16_t, Size>, COLOR_NB>        accumulation;
    std::array<std::array<std::int32_t, PSQTBuckets>, COLOR_NB> psqtAccumulation;
    std::array<bool, COLOR_NB>                                  computed = {};
};


// AccumulatorCaches struct provides per-thread accumulator caches, where each
// cache contains multiple entries for each of the possible king squares.
// When the accumulator needs to be refreshed, the cached entry is used to more
// efficiently update the accumulator, instead of rebuilding it from scratch.
// This idea, was first described by Luecx (author of Koivisto) and
// is commonly referred to as "Finny Tables".
struct AccumulatorCaches {

    template<typename Networks>
    AccumulatorCaches(const Networks& networks) {
        clear(networks);
    }

    template<IndexType Size>
    struct alignas(CacheLineSize) Cache {

        struct alignas(CacheLineSize) Entry {
            std::array<BiasType, Size>              accumulation;
            std::array<PSQTWeightType, PSQTBuckets> psqtAccumulation;
            std::array<Piece, SQUARE_NB>            pieces;
            Bitboard                                pieceBB;

            // To initialize a refresh entry, we set all its bitboards empty,
            // so we put the biases in the accumulation, without any weights on top
            void clear(const std::array<BiasType, Size>& biases) {
                accumulation = biases;
                std::memset(reinterpret_cast<std::byte*>(this) + offsetof(Entry, psqtAccumulation),
                            0, sizeof(Entry) - offsetof(Entry, psqtAccumulation));
            }
        };

        template<typename Network>
        void clear(const Network& network) {
            for (auto& entries1D : entries)
                for (auto& entry : entries1D)
                    entry.clear(network.featureTransformer.biases);
        }

        std::array<Entry, COLOR_NB>& operator[](Square sq) { return entries[sq]; }

        std::array<std::array<Entry, COLOR_NB>, SQUARE_NB> entries;
    };

    template<typename Networks>
    void clear(const Networks& networks) {
        big.clear(networks.big);
        small.clear(networks.small);
    }

    Cache<TransformedFeatureDimensionsBig>   big;
    Cache<TransformedFeatureDimensionsSmall> small;
};


template<typename FeatureSet, IndexType Dimensions>
struct AccumulatorStateSimple {
    Accumulator<Dimensions>       accumulator;
    typename FeatureSet::DiffType diff;

    auto& acc() noexcept { return accumulator; }

    const auto& acc() const noexcept { return accumulator; }

    void reset(const typename FeatureSet::DiffType& dp) noexcept {
        diff = dp;
        accumulator.computed.fill(false);
    }

    typename FeatureSet::DiffType& reset() noexcept {
        accumulator.computed.fill(false);
        return diff;
    }
};

template<IndexType Dimensions>
struct UpdateThreats {
    static constexpr std::size_t MaxSize = MAX_PLY + 1;

    auto&       latest() noexcept { return acc[size - 1]; }
    const auto& latest() const noexcept { return acc[size - 1]; }

    void reset_empty() {
        acc[0].reset({});
        size = 1;
    }

    auto& reset() {
        assert(size < MaxSize);
        acc[size].reset({});
        return acc[size++];
    }

    void pop() {
        assert(size > 1);
        size--;
    }

    std::size_t find_last_usable_accumulator(Color perspective) const noexcept;
    void        forward_update_incremental(
             Color                                 perspective,
             const Position&                       pos,
             const FeatureTransformer<Dimensions>& featureTransformer,
             const std::array<AccumulatorStateSimple<PSQFeatureSet, Dimensions>, MaxSize>& psqtAcc,
             const std::size_t                                                             begin) noexcept;
    void backward_update_incremental(Color                                 perspective,
                                     const Position&                       pos,
                                     const FeatureTransformer<Dimensions>& featureTransformer,
                                     const std::size_t                     end) noexcept;

    std::size_t                                                               size = 1;
    std::array<AccumulatorStateSimple<ThreatFeatureSet, Dimensions>, MaxSize> acc;
};

template<IndexType Dimensions>
struct UpdateHalfka {
    static constexpr std::size_t MaxSize = MAX_PLY + 1;

    auto&       latest() noexcept { return acc[size - 1]; }
    const auto& latest() const noexcept { return acc[size - 1]; }

    void reset_empty() {
        acc[0].reset({});
        size = 1;
    }

    auto& reset() {
        assert(size < MaxSize);
        acc[size].reset({});
        return acc[size++];
    }

    void pop() {
        assert(size > 1);
        size--;
    }

    std::size_t find_last_usable_accumulator(Color perspective) const noexcept;
    void        forward_update_incremental(Color                                 perspective,
                                           const Position&                       pos,
                                           const FeatureTransformer<Dimensions>& featureTransformer,
                                           const std::size_t                     begin) noexcept;

    void backward_update_incremental(Color                                 perspective,
                                     const Position&                       pos,
                                     const FeatureTransformer<Dimensions>& featureTransformer,
                                     const std::size_t                     end) noexcept;

    std::size_t                                                            size = 1;
    std::array<AccumulatorStateSimple<PSQFeatureSet, Dimensions>, MaxSize> acc;
};

struct BigNetworkAccumulator {

    void reset() noexcept;
    void push(const DirtyBoardData& dirtyBoardData) noexcept;
    void pop() noexcept;
    void evaluate(const Position&                                            pos,
                  const FeatureTransformer<TransformedFeatureDimensionsBig>& featureTransformer,
                  AccumulatorCaches::Cache<TransformedFeatureDimensionsBig>& cache) noexcept;

    UpdateHalfka<TransformedFeatureDimensionsBig>  psqt;
    UpdateThreats<TransformedFeatureDimensionsBig> threat;
};

struct SmallNetworkAccumulator {

    void reset() noexcept;
    void push(const DirtyPiece& dp) noexcept;
    void pop() noexcept;
    void evaluate(const Position&                                              pos,
                  const FeatureTransformer<TransformedFeatureDimensionsSmall>& featureTransformer,
                  AccumulatorCaches::Cache<TransformedFeatureDimensionsSmall>& cache) noexcept;


    UpdateHalfka<TransformedFeatureDimensionsSmall> psqt;
};

}  // namespace Stockfish::Eval::NNUE

#endif  // NNUE_ACCUMULATOR_NEW_H_INCLUDED
