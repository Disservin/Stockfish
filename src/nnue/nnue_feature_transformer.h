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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <utility>

#include "../position.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE {

using BiasType       = std::int16_t;
using WeightType     = std::int16_t;
using PSQTWeightType = std::int32_t;

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
#define VECTOR

static_assert(PSQTBuckets % 8 == 0,
              "Per feature PSQT values cannot be processed at granularity lower than 8 at a time.");

#ifdef USE_AVX512
using vec_t      = __m512i;
using psqt_vec_t = __m256i;
    #define vec_load(a) _mm512_load_si512(a)
    #define vec_store(a, b) _mm512_store_si512(a, b)
    #define vec_add_16(a, b) _mm512_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
    #define vec_mul_16(a, b) _mm512_mullo_epi16(a, b)
    #define vec_zero() _mm512_setzero_epi32()
    #define vec_set_16(a) _mm512_set1_epi16(a)
    #define vec_max_16(a, b) _mm512_max_epi16(a, b)
    #define vec_min_16(a, b) _mm512_min_epi16(a, b)
    // Inverse permuted at load time
    #define vec_msb_pack_16(a, b) \
        _mm512_packs_epi16(_mm512_srli_epi16(a, 7), _mm512_srli_epi16(b, 7))
    #define vec_load_psqt(a) _mm256_load_si256(a)
    #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
    #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
    #define vec_zero_psqt() _mm256_setzero_si256()
    #define NumRegistersSIMD 16
    #define MaxChunkSize 64

#elif USE_AVX2
using vec_t      = __m256i;
using psqt_vec_t = __m256i;
    #define vec_load(a) _mm256_load_si256(a)
    #define vec_store(a, b) _mm256_store_si256(a, b)
    #define vec_add_16(a, b) _mm256_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
    #define vec_mul_16(a, b) _mm256_mullo_epi16(a, b)
    #define vec_zero() _mm256_setzero_si256()
    #define vec_set_16(a) _mm256_set1_epi16(a)
    #define vec_max_16(a, b) _mm256_max_epi16(a, b)
    #define vec_min_16(a, b) _mm256_min_epi16(a, b)
    // Inverse permuted at load time
    #define vec_msb_pack_16(a, b) \
        _mm256_packs_epi16(_mm256_srli_epi16(a, 7), _mm256_srli_epi16(b, 7))
    #define vec_load_psqt(a) _mm256_load_si256(a)
    #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
    #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
    #define vec_zero_psqt() _mm256_setzero_si256()
    #define NumRegistersSIMD 16
    #define MaxChunkSize 32

#elif USE_SSE2
using vec_t      = __m128i;
using psqt_vec_t = __m128i;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) _mm_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm_sub_epi16(a, b)
    #define vec_mul_16(a, b) _mm_mullo_epi16(a, b)
    #define vec_zero() _mm_setzero_si128()
    #define vec_set_16(a) _mm_set1_epi16(a)
    #define vec_max_16(a, b) _mm_max_epi16(a, b)
    #define vec_min_16(a, b) _mm_min_epi16(a, b)
    #define vec_msb_pack_16(a, b) _mm_packs_epi16(_mm_srli_epi16(a, 7), _mm_srli_epi16(b, 7))
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) _mm_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm_sub_epi32(a, b)
    #define vec_zero_psqt() _mm_setzero_si128()
    #define NumRegistersSIMD (Is64Bit ? 16 : 8)
    #define MaxChunkSize 16

#elif USE_NEON
using vec_t      = int16x8_t;
using psqt_vec_t = int32x4_t;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) vaddq_s16(a, b)
    #define vec_sub_16(a, b) vsubq_s16(a, b)
    #define vec_mul_16(a, b) vmulq_s16(a, b)
    #define vec_zero() \
        vec_t { 0 }
    #define vec_set_16(a) vdupq_n_s16(a)
    #define vec_max_16(a, b) vmaxq_s16(a, b)
    #define vec_min_16(a, b) vminq_s16(a, b)
inline vec_t vec_msb_pack_16(vec_t a, vec_t b) {
    const int8x8_t  shifta    = vshrn_n_s16(a, 7);
    const int8x8_t  shiftb    = vshrn_n_s16(b, 7);
    const int8x16_t compacted = vcombine_s8(shifta, shiftb);
    return *reinterpret_cast<const vec_t*>(&compacted);
}
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) vaddq_s32(a, b)
    #define vec_sub_psqt_32(a, b) vsubq_s32(a, b)
    #define vec_zero_psqt() \
        psqt_vec_t { 0 }
    #define NumRegistersSIMD 16
    #define MaxChunkSize 16

#else
    #undef VECTOR

#endif


#ifdef VECTOR

    // Compute optimal SIMD register count for feature transformer accumulation.

    // We use __m* types as template arguments, which causes GCC to emit warnings
    // about losing some attribute information. This is irrelevant to us as we
    // only take their size, so the following pragma are harmless.
    #if defined(__GNUC__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif

template<typename SIMDRegisterType, typename LaneType, int NumLanes, int MaxRegisters>
static constexpr int BestRegisterCount() {
    #define RegisterSize sizeof(SIMDRegisterType)
    #define LaneSize sizeof(LaneType)

    static_assert(RegisterSize >= LaneSize);
    static_assert(MaxRegisters <= NumRegistersSIMD);
    static_assert(MaxRegisters > 0);
    static_assert(NumRegistersSIMD > 0);
    static_assert(RegisterSize % LaneSize == 0);
    static_assert((NumLanes * LaneSize) % RegisterSize == 0);

    const int ideal = (NumLanes * LaneSize) / RegisterSize;
    if (ideal <= MaxRegisters)
        return ideal;

    // Look for the largest divisor of the ideal register count that is smaller than MaxRegisters
    for (int divisor = MaxRegisters; divisor > 1; --divisor)
        if (ideal % divisor == 0)
            return divisor;

    return 1;
}
    #if defined(__GNUC__)
        #pragma GCC diagnostic pop
    #endif
#endif


// Input feature converter
template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
class FeatureTransformer {
   public:
    // Number of output dimensions for one side
    static constexpr IndexType HalfDimensions = TransformedFeatureDimensions;

   private:
#ifdef VECTOR
    static constexpr int NumRegs =
      BestRegisterCount<vec_t, WeightType, TransformedFeatureDimensions, NumRegistersSIMD>();
    static constexpr int NumPsqtRegs =
      BestRegisterCount<psqt_vec_t, PSQTWeightType, PSQTBuckets, NumRegistersSIMD>();

    static constexpr IndexType TileHeight     = NumRegs * sizeof(vec_t) / 2;
    static constexpr IndexType PsqtTileHeight = NumPsqtRegs * sizeof(psqt_vec_t) / 4;
    static_assert(HalfDimensions % TileHeight == 0, "TileHeight must divide HalfDimensions");
    static_assert(PSQTBuckets % PsqtTileHeight == 0, "PsqtTileHeight must divide PSQTBuckets");
#endif

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = FeatureSet::Dimensions;
    static constexpr IndexType OutputDimensions = HalfDimensions;

    // Size of forward propagation buffer
    static constexpr std::size_t BufferSize = OutputDimensions * sizeof(OutputType);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
        return FeatureSet::HashValue ^ (OutputDimensions * 2);
    }

    static constexpr void order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // _mm512_packs_epi16 ordering
        uint64_t tmp0 = v[2], tmp1 = v[3];
        v[2] = v[8], v[3] = v[9];
        v[8] = v[4], v[9] = v[5];
        v[4] = tmp0, v[5] = tmp1;
        tmp0 = v[6], tmp1 = v[7];
        v[6] = v[10], v[7] = v[11];
        v[10] = v[12], v[11] = v[13];
        v[12] = tmp0, v[13] = tmp1;
#elif defined(USE_AVX2)  // _mm256_packs_epi16 ordering
        std::swap(v[2], v[4]);
        std::swap(v[3], v[5]);
#endif
    }

    static constexpr void inverse_order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // Inverse _mm512_packs_epi16 ordering
        uint64_t tmp0 = v[2], tmp1 = v[3];
        v[2] = v[4], v[3] = v[5];
        v[4] = v[8], v[5] = v[9];
        v[8] = tmp0, v[9] = tmp1;
        tmp0 = v[6], tmp1 = v[7];
        v[6] = v[12], v[7] = v[13];
        v[12] = v[10], v[13] = v[11];
        v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // Inverse _mm256_packs_epi16 ordering
        std::swap(v[2], v[4]);
        std::swap(v[3], v[5]);
#endif
    }

    void permute_weights([[maybe_unused]] void (*order_fn)(uint64_t*)) const {
#if defined(USE_AVX2)
    #if defined(USE_AVX512)
        constexpr IndexType di = 16;
    #else
        constexpr IndexType di = 8;
    #endif
        uint64_t* b = reinterpret_cast<uint64_t*>(const_cast<BiasType*>(&biases[0]));
        for (IndexType i = 0; i < HalfDimensions * sizeof(BiasType) / sizeof(uint64_t); i += di)
            order_fn(&b[i]);

        for (IndexType j = 0; j < InputDimensions; ++j)
        {
            uint64_t* w =
              reinterpret_cast<uint64_t*>(const_cast<WeightType*>(&weights[j * HalfDimensions]));
            for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
                 i += di)
                order_fn(&w[i]);
        }
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {

        read_leb_128<BiasType>(stream, biases, HalfDimensions);
        read_leb_128<WeightType>(stream, weights, HalfDimensions * InputDimensions);
        read_leb_128<PSQTWeightType>(stream, psqtWeights, PSQTBuckets * InputDimensions);

        permute_weights(inverse_order_packs);
        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {

        permute_weights(order_packs);

        write_leb_128<BiasType>(stream, biases, HalfDimensions);
        write_leb_128<WeightType>(stream, weights, HalfDimensions * InputDimensions);
        write_leb_128<PSQTWeightType>(stream, psqtWeights, PSQTBuckets * InputDimensions);

        permute_weights(inverse_order_packs);
        return !stream.fail();
    }

    // Convert input features
    std::int32_t transform(const Position&                               pos,
                           AccumulatorCaches::CacheType<HalfDimensions>& cache,
                           OutputType*                                   output,
                           int                                           bucket,
                           bool                                          psqtOnly) const {
        update_accumulator<WHITE>(pos, cache, psqtOnly);
        update_accumulator<BLACK>(pos, cache, psqtOnly);

        const Color perspectives[2]  = {pos.side_to_move(), ~pos.side_to_move()};
        const auto& psqtAccumulation = (pos.state()->*accPtr).psqtAccumulation;
        const auto  psqt =
          (psqtAccumulation[perspectives[0]][bucket] - psqtAccumulation[perspectives[1]][bucket])
          / 2;

        if (psqtOnly)
            return psqt;

        const auto& accumulation = (pos.state()->*accPtr).accumulation;

        for (IndexType p = 0; p < 2; ++p)
        {
            const IndexType offset = (HalfDimensions / 2) * p;

#if defined(VECTOR)

            constexpr IndexType OutputChunkSize = MaxChunkSize;
            static_assert((HalfDimensions / 2) % OutputChunkSize == 0);
            constexpr IndexType NumOutputChunks = HalfDimensions / 2 / OutputChunkSize;

            const vec_t Zero = vec_zero();
            const vec_t One  = vec_set_16(127);

            const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0]));
            const vec_t* in1 =
              reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][HalfDimensions / 2]));
            vec_t* out = reinterpret_cast<vec_t*>(output + offset);

            for (IndexType j = 0; j < NumOutputChunks; ++j)
            {
                const vec_t sum0a = vec_max_16(vec_min_16(in0[j * 2 + 0], One), Zero);
                const vec_t sum0b = vec_max_16(vec_min_16(in0[j * 2 + 1], One), Zero);
                const vec_t sum1a = vec_max_16(vec_min_16(in1[j * 2 + 0], One), Zero);
                const vec_t sum1b = vec_max_16(vec_min_16(in1[j * 2 + 1], One), Zero);

                const vec_t pa = vec_mul_16(sum0a, sum1a);
                const vec_t pb = vec_mul_16(sum0b, sum1b);

                out[j] = vec_msb_pack_16(pa, pb);
            }

#else

            for (IndexType j = 0; j < HalfDimensions / 2; ++j)
            {
                BiasType sum0 = accumulation[static_cast<int>(perspectives[p])][j + 0];
                BiasType sum1 =
                  accumulation[static_cast<int>(perspectives[p])][j + HalfDimensions / 2];
                sum0               = std::clamp<BiasType>(sum0, 0, 127);
                sum1               = std::clamp<BiasType>(sum1, 0, 127);
                output[offset + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 128);
            }

#endif
        }

        return psqt;
    }  // end of function transform()

    void hint_common_access(const Position&                               pos,
                            AccumulatorCaches::CacheType<HalfDimensions>& cache,
                            bool                                          psqtOnly) const {
        hint_common_access_for_perspective<WHITE>(pos, cache, psqtOnly);
        hint_common_access_for_perspective<BLACK>(pos, cache, psqtOnly);
    }

   private:
    template<Color Perspective>
    [[nodiscard]] std::pair<StateInfo*, StateInfo*>
    try_find_computed_accumulator(const Position& pos, bool psqtOnly) const {
        // Look for a usable accumulator of an earlier position. We keep track
        // of the estimated gain in terms of features to be added/subtracted.
        StateInfo *st = pos.state(), *next = nullptr;
        int        gain = FeatureSet::refresh_cost(pos);
        while (st->previous
               && (!(st->*accPtr).computedPSQT[Perspective]
                   || (!psqtOnly && !(st->*accPtr).computed[Perspective])))
        {
            // This governs when a full feature refresh is needed and how many
            // updates are better than just one full refresh.
            if (FeatureSet::requires_refresh(st, Perspective)
                || (gain -= FeatureSet::update_cost(st) + 1) < 0)
                break;
            next = st;
            st   = st->previous;
        }
        return {st, next};
    }

    // NOTE: The parameter states_to_update is an array of position states, ending with nullptr.
    //       All states must be sequential, that is states_to_update[i] must either be reachable
    //       by repeatedly applying ->previous from states_to_update[i+1] or
    //       states_to_update[i] == nullptr.
    //       computed_st must be reachable by repeatedly applying ->previous on
    //       states_to_update[0], if not nullptr.
    template<Color Perspective, size_t N>
    void update_accumulator_incremental(const Position& pos,
                                        StateInfo*      computed_st,
                                        StateInfo*      states_to_update[N],
                                        bool            psqtOnly) const {
        static_assert(N > 0);
        assert(states_to_update[N - 1] == nullptr);

#ifdef VECTOR
        // Gcc-10.2 unnecessarily spills AVX2 registers if this array
        // is defined in the VECTOR code below, once in each branch
        vec_t      acc[NumRegs];
        psqt_vec_t psqt[NumPsqtRegs];
#endif

        if (states_to_update[0] == nullptr)
            return;

        // Update incrementally going back through states_to_update.

        // Gather all features to be updated.
        const Square ksq = pos.square<KING>(Perspective);

        // The size must be enough to contain the largest possible update.
        // That might depend on the feature set and generally relies on the
        // feature set's update cost calculation to be correct and never allow
        // updates with more added/removed features than MaxActiveDimensions.
        FeatureSet::IndexList removed[N - 1], added[N - 1];

        {
            int i =
              N
              - 2;  // Last potential state to update. Skip last element because it must be nullptr.
            while (states_to_update[i] == nullptr)
                --i;

            StateInfo* st2 = states_to_update[i];

            for (; i >= 0; --i)
            {
                (states_to_update[i]->*accPtr).computed[Perspective]     = !psqtOnly;
                (states_to_update[i]->*accPtr).computedPSQT[Perspective] = true;

                const StateInfo* end_state = i == 0 ? computed_st : states_to_update[i - 1];

                for (; st2 != end_state; st2 = st2->previous)
                    FeatureSet::append_changed_indices<Perspective>(ksq, st2->dirtyPiece,
                                                                    removed[i], added[i]);
            }
        }

        StateInfo* st = computed_st;

        // Now update the accumulators listed in states_to_update[], where the last element is a sentinel.
#ifdef VECTOR

        if (states_to_update[1] == nullptr && (removed[0].size() == 1 || removed[0].size() == 2)
            && added[0].size() == 1)
        {
            assert(states_to_update[0]);

            if (!psqtOnly)
            {
                auto accIn =
                  reinterpret_cast<const vec_t*>(&(st->*accPtr).accumulation[Perspective][0]);
                auto accOut = reinterpret_cast<vec_t*>(
                  &(states_to_update[0]->*accPtr).accumulation[Perspective][0]);

                const IndexType offsetR0 = HalfDimensions * removed[0][0];
                auto            columnR0 = reinterpret_cast<const vec_t*>(&weights[offsetR0]);
                const IndexType offsetA  = HalfDimensions * added[0][0];
                auto            columnA  = reinterpret_cast<const vec_t*>(&weights[offsetA]);

                if (removed[0].size() == 1)
                {
                    for (IndexType k = 0; k < HalfDimensions * sizeof(std::int16_t) / sizeof(vec_t);
                         ++k)
                        accOut[k] = vec_add_16(vec_sub_16(accIn[k], columnR0[k]), columnA[k]);
                }
                else
                {
                    const IndexType offsetR1 = HalfDimensions * removed[0][1];
                    auto            columnR1 = reinterpret_cast<const vec_t*>(&weights[offsetR1]);

                    for (IndexType k = 0; k < HalfDimensions * sizeof(std::int16_t) / sizeof(vec_t);
                         ++k)
                        accOut[k] = vec_sub_16(vec_add_16(accIn[k], columnA[k]),
                                               vec_add_16(columnR0[k], columnR1[k]));
                }
            }

            auto accPsqtIn =
              reinterpret_cast<const psqt_vec_t*>(&(st->*accPtr).psqtAccumulation[Perspective][0]);
            auto accPsqtOut = reinterpret_cast<psqt_vec_t*>(
              &(states_to_update[0]->*accPtr).psqtAccumulation[Perspective][0]);

            const IndexType offsetPsqtR0 = PSQTBuckets * removed[0][0];
            auto columnPsqtR0 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtR0]);
            const IndexType offsetPsqtA = PSQTBuckets * added[0][0];
            auto columnPsqtA = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtA]);

            if (removed[0].size() == 1)
            {
                for (std::size_t k = 0; k < PSQTBuckets * sizeof(std::int32_t) / sizeof(psqt_vec_t);
                     ++k)
                    accPsqtOut[k] = vec_add_psqt_32(vec_sub_psqt_32(accPsqtIn[k], columnPsqtR0[k]),
                                                    columnPsqtA[k]);
            }
            else
            {
                const IndexType offsetPsqtR1 = PSQTBuckets * removed[0][1];
                auto columnPsqtR1 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offsetPsqtR1]);

                for (std::size_t k = 0; k < PSQTBuckets * sizeof(std::int32_t) / sizeof(psqt_vec_t);
                     ++k)
                    accPsqtOut[k] =
                      vec_sub_psqt_32(vec_add_psqt_32(accPsqtIn[k], columnPsqtA[k]),
                                      vec_add_psqt_32(columnPsqtR0[k], columnPsqtR1[k]));
            }
        }
        else
        {
            if (!psqtOnly)
                for (IndexType j = 0; j < HalfDimensions / TileHeight; ++j)
                {
                    // Load accumulator
                    auto accTileIn = reinterpret_cast<const vec_t*>(
                      &(st->*accPtr).accumulation[Perspective][j * TileHeight]);
                    for (IndexType k = 0; k < NumRegs; ++k)
                        acc[k] = vec_load(&accTileIn[k]);

                    for (IndexType i = 0; states_to_update[i]; ++i)
                    {
                        // Difference calculation for the deactivated features
                        for (const auto index : removed[i])
                        {
                            const IndexType offset = HalfDimensions * index + j * TileHeight;
                            auto column = reinterpret_cast<const vec_t*>(&weights[offset]);
                            for (IndexType k = 0; k < NumRegs; ++k)
                                acc[k] = vec_sub_16(acc[k], column[k]);
                        }

                        // Difference calculation for the activated features
                        for (const auto index : added[i])
                        {
                            const IndexType offset = HalfDimensions * index + j * TileHeight;
                            auto column = reinterpret_cast<const vec_t*>(&weights[offset]);
                            for (IndexType k = 0; k < NumRegs; ++k)
                                acc[k] = vec_add_16(acc[k], column[k]);
                        }

                        // Store accumulator
                        auto accTileOut =
                          reinterpret_cast<vec_t*>(&(states_to_update[i]->*accPtr)
                                                      .accumulation[Perspective][j * TileHeight]);
                        for (IndexType k = 0; k < NumRegs; ++k)
                            vec_store(&accTileOut[k], acc[k]);
                    }
                }

            for (IndexType j = 0; j < PSQTBuckets / PsqtTileHeight; ++j)
            {
                // Load accumulator
                auto accTilePsqtIn = reinterpret_cast<const psqt_vec_t*>(
                  &(st->*accPtr).psqtAccumulation[Perspective][j * PsqtTileHeight]);
                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] = vec_load_psqt(&accTilePsqtIn[k]);

                for (IndexType i = 0; states_to_update[i]; ++i)
                {
                    // Difference calculation for the deactivated features
                    for (const auto index : removed[i])
                    {
                        const IndexType offset = PSQTBuckets * index + j * PsqtTileHeight;
                        auto columnPsqt = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);
                        for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                            psqt[k] = vec_sub_psqt_32(psqt[k], columnPsqt[k]);
                    }

                    // Difference calculation for the activated features
                    for (const auto index : added[i])
                    {
                        const IndexType offset = PSQTBuckets * index + j * PsqtTileHeight;
                        auto columnPsqt = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);
                        for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                            psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
                    }

                    // Store accumulator
                    auto accTilePsqtOut = reinterpret_cast<psqt_vec_t*>(
                      &(states_to_update[i]->*accPtr)
                         .psqtAccumulation[Perspective][j * PsqtTileHeight]);
                    for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                        vec_store_psqt(&accTilePsqtOut[k], psqt[k]);
                }
            }
        }
#else
        for (IndexType i = 0; states_to_update[i]; ++i)
        {
            if (!psqtOnly)
                std::memcpy((states_to_update[i]->*accPtr).accumulation[Perspective],
                            (st->*accPtr).accumulation[Perspective],
                            HalfDimensions * sizeof(BiasType));

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                (states_to_update[i]->*accPtr).psqtAccumulation[Perspective][k] =
                  (st->*accPtr).psqtAccumulation[Perspective][k];

            st = states_to_update[i];

            // Difference calculation for the deactivated features
            for (const auto index : removed[i])
            {
                if (!psqtOnly)
                {
                    const IndexType offset = HalfDimensions * index;
                    for (IndexType j = 0; j < HalfDimensions; ++j)
                        (st->*accPtr).accumulation[Perspective][j] -= weights[offset + j];
                }

                for (std::size_t k = 0; k < PSQTBuckets; ++k)
                    (st->*accPtr).psqtAccumulation[Perspective][k] -=
                      psqtWeights[index * PSQTBuckets + k];
            }

            // Difference calculation for the activated features
            for (const auto index : added[i])
            {
                if (!psqtOnly)
                {
                    const IndexType offset = HalfDimensions * index;
                    for (IndexType j = 0; j < HalfDimensions; ++j)
                        (st->*accPtr).accumulation[Perspective][j] += weights[offset + j];
                }

                for (std::size_t k = 0; k < PSQTBuckets; ++k)
                    (st->*accPtr).psqtAccumulation[Perspective][k] +=
                      psqtWeights[index * PSQTBuckets + k];
            }
        }
#endif
    }

    template<Color Perspective>
    void
    update_accumulator_refresh_cache(const Position&                               pos,
                                     AccumulatorCaches::CacheType<HalfDimensions>& cache) const {

        Square ksq = pos.square<KING>(Perspective);

        auto& entry = (*cache)[ksq];

        auto& accumulator                     = pos.state()->*accPtr;
        accumulator.computed[Perspective]     = true;
        accumulator.computedPSQT[Perspective] = true;

        FeatureSet::IndexList removed, added;
        for (Color c : {WHITE, BLACK})
        {
            for (PieceType pt = PAWN; pt <= KING; ++pt)
            {
                const Piece    piece = make_piece(c, pt);
                const Bitboard oldBB =
                  entry.byColorBB[Perspective][c] & entry.byTypeBB[Perspective][pt];
                const Bitboard newBB    = pos.pieces(c, pt);
                Bitboard       toRemove = oldBB & ~newBB;
                Bitboard       toAdd    = newBB & ~oldBB;

                while (toRemove)
                {
                    Square sq = pop_lsb(toRemove);
                    removed.push_back(FeatureSet::make_index<Perspective>(sq, piece, ksq));
                }
                while (toAdd)
                {
                    Square sq = pop_lsb(toAdd);
                    added.push_back(FeatureSet::make_index<Perspective>(sq, piece, ksq));
                }
            }
        }

#ifdef VECTOR
        vec_t      acc[NumRegs];
        psqt_vec_t psqt[NumPsqtRegs];

        for (IndexType j = 0; j < HalfDimensions / TileHeight; ++j)
        {
            auto entryTile =
              reinterpret_cast<vec_t*>(&entry.accumulation[Perspective][j * TileHeight]);
            for (IndexType k = 0; k < NumRegs; ++k)
                acc[k] = entryTile[k];

            for (int i = 0; i < int(added.size()); ++i)
            {
                IndexType       index  = added[i];
                const IndexType offset = HalfDimensions * index + j * TileHeight;
                auto            column = reinterpret_cast<const vec_t*>(&weights[offset]);

                for (unsigned k = 0; k < NumRegs; ++k)
                    acc[k] = vec_add_16(acc[k], column[k]);
            }
            for (int i = 0; i < int(removed.size()); ++i)
            {
                IndexType       index  = removed[i];
                const IndexType offset = HalfDimensions * index + j * TileHeight;
                auto            column = reinterpret_cast<const vec_t*>(&weights[offset]);

                for (unsigned k = 0; k < NumRegs; ++k)
                    acc[k] = vec_sub_16(acc[k], column[k]);
            }

            for (IndexType k = 0; k < NumRegs; k++)
                vec_store(&entryTile[k], acc[k]);
        }

        for (IndexType j = 0; j < PSQTBuckets / PsqtTileHeight; ++j)
        {
            auto entryTilePsqt = reinterpret_cast<psqt_vec_t*>(
              &entry.psqtAccumulation[Perspective][j * PsqtTileHeight]);
            for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                psqt[k] = entryTilePsqt[k];

            for (int i = 0; i < int(added.size()); ++i)
            {
                IndexType       index  = added[i];
                const IndexType offset = PSQTBuckets * index + j * PsqtTileHeight;
                auto columnPsqt        = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
            }
            for (int i = 0; i < int(removed.size()); ++i)
            {
                IndexType       index  = removed[i];
                const IndexType offset = PSQTBuckets * index + j * PsqtTileHeight;
                auto columnPsqt        = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] = vec_sub_psqt_32(psqt[k], columnPsqt[k]);
            }

            for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                vec_store_psqt(&entryTilePsqt[k], psqt[k]);
        }

#else

        for (const auto index : added)
        {
            const IndexType offset = HalfDimensions * index;
            for (IndexType j = 0; j < HalfDimensions; ++j)
                entry.accumulation[Perspective][j] += weights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                entry.psqtAccumulation[Perspective][k] += psqtWeights[index * PSQTBuckets + k];
        }
        for (const auto index : removed)
        {
            const IndexType offset = HalfDimensions * index;
            for (IndexType j = 0; j < HalfDimensions; ++j)
                entry.accumulation[Perspective][j] -= weights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                entry.psqtAccumulation[Perspective][k] -= psqtWeights[index * PSQTBuckets + k];
        }

#endif

        // The accumulator of the refresh entry has been updated.
        // Now copy its content to the actual accumulator we were refreshing

        std::memcpy(accumulator.psqtAccumulation[Perspective], entry.psqtAccumulation[Perspective],
                    sizeof(int32_t) * PSQTBuckets);

        std::memcpy(accumulator.accumulation[Perspective], entry.accumulation[Perspective],
                    sizeof(BiasType) * HalfDimensions);

        for (Color c : {WHITE, BLACK})
            entry.byColorBB[Perspective][c] = pos.pieces(c);

        for (PieceType pt = PAWN; pt <= KING; ++pt)
            entry.byTypeBB[Perspective][pt] = pos.pieces(pt);
    }

    template<Color Perspective>
    void update_accumulator_refresh(const Position&                               pos,
                                    AccumulatorCaches::CacheType<HalfDimensions>& cache,
                                    bool                                          psqtOnly) const {

        // When we are refreshing the accumulator of the big net,
        // redirect to the version of refresh that uses the refresh table.
        // Using the cache for the small net is not beneficial.
        if (HalfDimensions == Eval::NNUE::TransformedFeatureDimensionsBig)
        {
            update_accumulator_refresh_cache<Perspective>(pos, cache);
            return;
        }

#ifdef VECTOR
        // Gcc-10.2 unnecessarily spills AVX2 registers if this array
        // is defined in the VECTOR code below, once in each branch
        vec_t      acc[NumRegs];
        psqt_vec_t psqt[NumPsqtRegs];
#endif

        // Refresh the accumulator
        // Could be extracted to a separate function because it's done in 2 places,
        // but it's unclear if compilers would correctly handle register allocation.
        auto& accumulator                     = pos.state()->*accPtr;
        accumulator.computed[Perspective]     = !psqtOnly;
        accumulator.computedPSQT[Perspective] = true;
        FeatureSet::IndexList active;
        FeatureSet::append_active_indices<Perspective>(pos, active);

#ifdef VECTOR
        if (!psqtOnly)
            for (IndexType j = 0; j < HalfDimensions / TileHeight; ++j)
            {
                auto biasesTile = reinterpret_cast<const vec_t*>(&biases[j * TileHeight]);
                for (IndexType k = 0; k < NumRegs; ++k)
                    acc[k] = biasesTile[k];

                int i = 0;
                for (; i < int(active.size()) - 1; i += 2)
                {
                    IndexType       index0  = active[i];
                    IndexType       index1  = active[i + 1];
                    const IndexType offset0 = HalfDimensions * index0 + j * TileHeight;
                    const IndexType offset1 = HalfDimensions * index1 + j * TileHeight;
                    auto            column0 = reinterpret_cast<const vec_t*>(&weights[offset0]);
                    auto            column1 = reinterpret_cast<const vec_t*>(&weights[offset1]);

                    for (unsigned k = 0; k < NumRegs; ++k)
                        acc[k] = vec_add_16(acc[k], vec_add_16(column0[k], column1[k]));
                }
                for (; i < int(active.size()); ++i)
                {
                    IndexType       index  = active[i];
                    const IndexType offset = HalfDimensions * index + j * TileHeight;
                    auto            column = reinterpret_cast<const vec_t*>(&weights[offset]);

                    for (unsigned k = 0; k < NumRegs; ++k)
                        acc[k] = vec_add_16(acc[k], column[k]);
                }

                auto accTile =
                  reinterpret_cast<vec_t*>(&accumulator.accumulation[Perspective][j * TileHeight]);
                for (unsigned k = 0; k < NumRegs; k++)
                    vec_store(&accTile[k], acc[k]);
            }

        for (IndexType j = 0; j < PSQTBuckets / PsqtTileHeight; ++j)
        {
            for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                psqt[k] = vec_zero_psqt();

            int i = 0;
            for (; i < int(active.size()) - 1; i += 2)
            {
                IndexType       index0  = active[i];
                IndexType       index1  = active[i + 1];
                const IndexType offset0 = PSQTBuckets * index0 + j * PsqtTileHeight;
                const IndexType offset1 = PSQTBuckets * index1 + j * PsqtTileHeight;
                auto columnPsqt0 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset0]);
                auto columnPsqt1 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset1]);

                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] =
                      vec_add_psqt_32(psqt[k], vec_add_psqt_32(columnPsqt0[k], columnPsqt1[k]));
            }
            for (; i < int(active.size()); ++i)
            {
                IndexType       index  = active[i];
                const IndexType offset = PSQTBuckets * index + j * PsqtTileHeight;
                auto columnPsqt        = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
            }

            auto accTilePsqt = reinterpret_cast<psqt_vec_t*>(
              &accumulator.psqtAccumulation[Perspective][j * PsqtTileHeight]);
            for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                vec_store_psqt(&accTilePsqt[k], psqt[k]);
        }

#else
        if (!psqtOnly)
            std::memcpy(accumulator.accumulation[Perspective], biases,
                        HalfDimensions * sizeof(BiasType));

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            accumulator.psqtAccumulation[Perspective][k] = 0;

        for (const auto index : active)
        {
            if (!psqtOnly)
            {
                const IndexType offset = HalfDimensions * index;
                for (IndexType j = 0; j < HalfDimensions; ++j)
                    accumulator.accumulation[Perspective][j] += weights[offset + j];
            }

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                accumulator.psqtAccumulation[Perspective][k] +=
                  psqtWeights[index * PSQTBuckets + k];
        }
#endif
    }

    template<Color Perspective>
    void hint_common_access_for_perspective(const Position&                               pos,
                                            AccumulatorCaches::CacheType<HalfDimensions>& cache,
                                            bool psqtOnly) const {

        // Works like update_accumulator, but performs less work.
        // Updates ONLY the accumulator for pos.

        // Look for a usable accumulator of an earlier position. We keep track
        // of the estimated gain in terms of features to be added/subtracted.
        // Fast early exit.
        if ((pos.state()->*accPtr).computed[Perspective]
            || (psqtOnly && (pos.state()->*accPtr).computedPSQT[Perspective]))
            return;

        auto [oldest_st, _] = try_find_computed_accumulator<Perspective>(pos, psqtOnly);

        if ((oldest_st->*accPtr).computed[Perspective]
            || (psqtOnly && (oldest_st->*accPtr).computedPSQT[Perspective]))
        {
            // Only update current position accumulator to minimize work.
            StateInfo* states_to_update[2] = {pos.state(), nullptr};
            update_accumulator_incremental<Perspective, 2>(pos, oldest_st, states_to_update,
                                                           psqtOnly);
        }
        else
            update_accumulator_refresh<Perspective>(pos, cache, psqtOnly);
    }

    template<Color Perspective>
    void update_accumulator(const Position&                               pos,
                            AccumulatorCaches::CacheType<HalfDimensions>& cache,
                            bool                                          psqtOnly) const {

        auto [oldest_st, next] = try_find_computed_accumulator<Perspective>(pos, psqtOnly);

        if ((oldest_st->*accPtr).computed[Perspective]
            || (psqtOnly && (oldest_st->*accPtr).computedPSQT[Perspective]))
        {
            if (next == nullptr)
                return;

            // Now update the accumulators listed in states_to_update[], where the last element is a sentinel.
            // Currently we update 2 accumulators.
            //     1. for the current position
            //     2. the next accumulator after the computed one
            // The heuristic may change in the future.
            StateInfo* states_to_update[3] = {next, next == pos.state() ? nullptr : pos.state(),
                                              nullptr};

            update_accumulator_incremental<Perspective, 3>(pos, oldest_st, states_to_update,
                                                           psqtOnly);
        }
        else
            update_accumulator_refresh<Perspective>(pos, cache, psqtOnly);
    }

    template<IndexType Size>
    friend struct AccumulatorCaches::Cache;

    alignas(CacheLineSize) BiasType biases[HalfDimensions];
    alignas(CacheLineSize) WeightType weights[HalfDimensions * InputDimensions];
    alignas(CacheLineSize) PSQTWeightType psqtWeights[InputDimensions * PSQTBuckets];
};

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
