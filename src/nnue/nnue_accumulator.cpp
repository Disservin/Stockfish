/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

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

#include "nnue_accumulator_new.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <new>
#include <type_traits>

#include "../bitboard.h"
#include "../misc.h"
#include "../position.h"
#include "nnue_common.h"
#include "nnue_feature_transformer.h"
#include "simd.h"

namespace Stockfish::Eval::NNUE {

using namespace SIMD;

namespace {

template<typename VectorWrapper,
         IndexType Width,
         UpdateOperation... ops,
         typename ElementType,
         typename... Ts,
         std::enable_if_t<is_all_same_v<ElementType, Ts...>, bool> = true>
void fused_row_reduce(const ElementType* in, ElementType* out, const Ts* const... rows) {
    constexpr IndexType size = Width * sizeof(ElementType) / sizeof(typename VectorWrapper::type);

    auto* vecIn  = reinterpret_cast<const typename VectorWrapper::type*>(in);
    auto* vecOut = reinterpret_cast<typename VectorWrapper::type*>(out);

    for (IndexType i = 0; i < size; ++i)
        vecOut[i] = fused<VectorWrapper, ops...>(
          vecIn[i], reinterpret_cast<const typename VectorWrapper::type*>(rows)[i]...);
}

template<typename FeatureSet, IndexType Dimensions>
struct AccumulatorUpdateContext {
    Color                                                 perspective;
    const FeatureTransformer<Dimensions>&                 featureTransformer;
    const AccumulatorStateSimple<FeatureSet, Dimensions>& from;
    AccumulatorStateSimple<FeatureSet, Dimensions>&       to;

    AccumulatorUpdateContext(Color                                                 persp,
                             const FeatureTransformer<Dimensions>&                 ft,
                             const AccumulatorStateSimple<FeatureSet, Dimensions>& accF,
                             AccumulatorStateSimple<FeatureSet, Dimensions>&       accT) noexcept :
        perspective{persp},
        featureTransformer{ft},
        from{accF},
        to{accT} {}

    template<UpdateOperation... ops,
             typename... Ts,
             std::enable_if_t<is_all_same_v<IndexType, Ts...>, bool> = true>
    void apply(const Ts... indices) {
        auto to_weight_vector = [&](const IndexType index) {
            return &featureTransformer.weights[index * Dimensions];
        };

        auto to_psqt_weight_vector = [&](const IndexType index) {
            return &featureTransformer.psqtWeights[index * PSQTBuckets];
        };

        fused_row_reduce<Vec16Wrapper, Dimensions, ops...>(
          from.acc().accumulation[perspective].data(), to.acc().accumulation[perspective].data(),
          to_weight_vector(indices)...);

        fused_row_reduce<Vec32Wrapper, PSQTBuckets, ops...>(
          from.acc().psqtAccumulation[perspective].data(),
          to.acc().psqtAccumulation[perspective].data(), to_psqt_weight_vector(indices)...);
    }

    void apply(const typename FeatureSet::IndexList& added,
               const typename FeatureSet::IndexList& removed) {
        const auto& fromAcc = from.acc().accumulation[perspective];
        auto&       toAcc   = to.acc().accumulation[perspective];

        const auto& fromPsqtAcc = from.acc().psqtAccumulation[perspective];
        auto&       toPsqtAcc   = to.acc().psqtAccumulation[perspective];

#ifdef VECTOR
        using Tiling = SIMDTiling<Dimensions, Dimensions, PSQTBuckets>;
        vec_t      accVec[Tiling::NumRegs];
        psqt_vec_t psqtVec[Tiling::NumPsqtRegs];

        const auto* threatWeights = &featureTransformer.threatWeights[0];

        for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
        {
            auto* fromTile = reinterpret_cast<const vec_t*>(&fromAcc[j * Tiling::TileHeight]);
            auto* toTile   = reinterpret_cast<vec_t*>(&toAcc[j * Tiling::TileHeight]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                accVec[k] = fromTile[k];

            for (int i = 0; i < removed.ssize(); ++i)
            {
                size_t       index  = removed[i];
                const size_t offset = Dimensions * index;
                auto*        column = reinterpret_cast<const vec_i8_t*>(&threatWeights[offset]);

    #ifdef USE_NEON
                for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
                {
                    accVec[k]     = vec_sub_16(accVec[k], vmovl_s8(vget_low_s8(column[k / 2])));
                    accVec[k + 1] = vec_sub_16(accVec[k + 1], vmovl_high_s8(column[k / 2]));
                }
    #else
                for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                    accVec[k] = vec_sub_16(accVec[k], vec_convert_8_16(column[k]));
    #endif
            }

            for (int i = 0; i < added.ssize(); ++i)
            {
                size_t       index  = added[i];
                const size_t offset = Dimensions * index;
                auto*        column = reinterpret_cast<const vec_i8_t*>(&threatWeights[offset]);

    #ifdef USE_NEON
                for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
                {
                    accVec[k]     = vec_add_16(accVec[k], vmovl_s8(vget_low_s8(column[k / 2])));
                    accVec[k + 1] = vec_add_16(accVec[k + 1], vmovl_high_s8(column[k / 2]));
                }
    #else
                for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                    accVec[k] = vec_add_16(accVec[k], vec_convert_8_16(column[k]));
    #endif
            }

            for (IndexType k = 0; k < Tiling::NumRegs; k++)
                vec_store(&toTile[k], accVec[k]);

            threatWeights += Tiling::TileHeight;
        }

        for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
        {
            auto* fromTilePsqt =
              reinterpret_cast<const psqt_vec_t*>(&fromPsqtAcc[j * Tiling::PsqtTileHeight]);
            auto* toTilePsqt =
              reinterpret_cast<psqt_vec_t*>(&toPsqtAcc[j * Tiling::PsqtTileHeight]);

            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqtVec[k] = fromTilePsqt[k];

            for (int i = 0; i < removed.ssize(); ++i)
            {
                size_t       index      = removed[i];
                const size_t offset     = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
                auto*        columnPsqt = reinterpret_cast<const psqt_vec_t*>(
                  &featureTransformer.threatPsqtWeights[offset]);

                for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                    psqtVec[k] = vec_sub_psqt_32(psqtVec[k], columnPsqt[k]);
            }

            for (int i = 0; i < added.ssize(); ++i)
            {
                size_t       index      = added[i];
                const size_t offset     = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
                auto*        columnPsqt = reinterpret_cast<const psqt_vec_t*>(
                  &featureTransformer.threatPsqtWeights[offset]);

                for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                    psqtVec[k] = vec_add_psqt_32(psqtVec[k], columnPsqt[k]);
            }

            for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
                vec_store_psqt(&toTilePsqt[k], psqtVec[k]);
        }

#else

        toAcc     = fromAcc;
        toPsqtAcc = fromPsqtAcc;

        for (const auto index : removed)
        {
            const IndexType offset = Dimensions * index;

            for (IndexType j = 0; j < Dimensions; ++j)
                toAcc[j] -= featureTransformer.threatWeights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                toPsqtAcc[k] -= featureTransformer.threatPsqtWeights[index * PSQTBuckets + k];
        }

        for (const auto index : added)
        {
            const IndexType offset = Dimensions * index;

            for (IndexType j = 0; j < Dimensions; ++j)
                toAcc[j] += featureTransformer.threatWeights[offset + j];

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                toPsqtAcc[k] += featureTransformer.threatPsqtWeights[index * PSQTBuckets + k];
        }

#endif
    }
};

template<typename FeatureSet, IndexType Dimensions>
auto make_accumulator_update_context(
  Color                                                 perspective,
  const FeatureTransformer<Dimensions>&                 featureTransformer,
  const AccumulatorStateSimple<FeatureSet, Dimensions>& accumulatorFrom,
  AccumulatorStateSimple<FeatureSet, Dimensions>&       accumulatorTo) noexcept {
    return AccumulatorUpdateContext<FeatureSet, Dimensions>{perspective, featureTransformer,
                                                            accumulatorFrom, accumulatorTo};
}

template<IndexType Dimensions>
void double_inc_update(Color                                                    perspective,
                       const FeatureTransformer<Dimensions>&                    featureTransformer,
                       const Square                                             ksq,
                       AccumulatorStateSimple<PSQFeatureSet, Dimensions>&       middle_state,
                       AccumulatorStateSimple<PSQFeatureSet, Dimensions>&       target_state,
                       const AccumulatorStateSimple<PSQFeatureSet, Dimensions>& computed) {

    assert(computed.acc().computed[perspective]);
    assert(!middle_state.acc().computed[perspective]);
    assert(!target_state.acc().computed[perspective]);

    PSQFeatureSet::IndexList removed, added;
    PSQFeatureSet::append_changed_indices(perspective, ksq, middle_state.diff, removed, added);
    assert(added.size() < 2);
    PSQFeatureSet::append_changed_indices(perspective, ksq, target_state.diff, removed, added);

    assert(added.size() == 1);
    assert(removed.size() == 2 || removed.size() == 3);

    // Workaround compiler warning for uninitialized variables.
    sf_assume(added.size() == 1);
    sf_assume(removed.size() == 2 || removed.size() == 3);

    auto doubleContext =
      make_accumulator_update_context(perspective, featureTransformer, computed, target_state);

    if (removed.size() == 2)
        doubleContext.template apply<Add, Sub, Sub>(added[0], removed[0], removed[1]);
    else
    {
        assert(removed.size() == 3);
        doubleContext.template apply<Add, Sub, Sub, Sub>(added[0], removed[0], removed[1],
                                                         removed[2]);
    }

    target_state.acc().computed[perspective] = true;
}

template<IndexType Dimensions>
void double_inc_update(Color                                                 perspective,
                       const FeatureTransformer<Dimensions>&                 featureTransformer,
                       const Square                                          ksq,
                       AccumulatorStateSimple<ThreatFeatureSet, Dimensions>& middle_state,
                       AccumulatorStateSimple<ThreatFeatureSet, Dimensions>& target_state,
                       const AccumulatorStateSimple<ThreatFeatureSet, Dimensions>& computed,
                       const DirtyPiece&                                           dp2) {

    assert(computed.acc().computed[perspective]);
    assert(!middle_state.acc().computed[perspective]);
    assert(!target_state.acc().computed[perspective]);

    ThreatFeatureSet::FusedUpdateData fusedData;
    fusedData.dp2removed = dp2.remove_sq;

    ThreatFeatureSet::IndexList removed, added;
    ThreatFeatureSet::append_changed_indices(perspective, ksq, middle_state.diff, removed, added,
                                             &fusedData, true);
    ThreatFeatureSet::append_changed_indices(perspective, ksq, target_state.diff, removed, added,
                                             &fusedData, false);

    auto updateContext =
      make_accumulator_update_context(perspective, featureTransformer, computed, target_state);

    updateContext.apply(added, removed);

    target_state.acc().computed[perspective] = true;
}

template<bool Forward, typename FeatureSet, IndexType Dimensions>
void update_accumulator_incremental(
  Color                                                 perspective,
  const FeatureTransformer<Dimensions>&                 featureTransformer,
  const Square                                          ksq,
  AccumulatorStateSimple<FeatureSet, Dimensions>&       target_state,
  const AccumulatorStateSimple<FeatureSet, Dimensions>& computed) {

    assert(computed.acc().computed[perspective]);
    assert(!target_state.acc().computed[perspective]);

    typename FeatureSet::IndexList removed, added;
    if constexpr (Forward)
        FeatureSet::append_changed_indices(perspective, ksq, target_state.diff, removed, added);
    else
        FeatureSet::append_changed_indices(perspective, ksq, computed.diff, added, removed);

    auto updateContext =
      make_accumulator_update_context(perspective, featureTransformer, computed, target_state);

    if constexpr (std::is_same_v<FeatureSet, ThreatFeatureSet>)
        updateContext.apply(added, removed);
    else
    {
        assert(added.size() == 1 || added.size() == 2);
        assert(removed.size() == 1 || removed.size() == 2);
        assert((Forward && added.size() <= removed.size())
               || (!Forward && added.size() >= removed.size()));

        sf_assume(added.size() == 1 || added.size() == 2);
        sf_assume(removed.size() == 1 || removed.size() == 2);

        if ((Forward && removed.size() == 1) || (!Forward && added.size() == 1))
        {
            assert(added.size() == 1 && removed.size() == 1);
            updateContext.template apply<Add, Sub>(added[0], removed[0]);
        }
        else if (Forward && added.size() == 1)
        {
            assert(removed.size() == 2);
            updateContext.template apply<Add, Sub, Sub>(added[0], removed[0], removed[1]);
        }
        else if (!Forward && removed.size() == 1)
        {
            assert(added.size() == 2);
            updateContext.template apply<Add, Add, Sub>(added[0], added[1], removed[0]);
        }
        else
        {
            assert(added.size() == 2 && removed.size() == 2);
            updateContext.template apply<Add, Add, Sub, Sub>(added[0], added[1], removed[0],
                                                             removed[1]);
        }
    }

    target_state.acc().computed[perspective] = true;
}

}  // namespace

template<IndexType Dimensions>
std::size_t
UpdateHalfka<Dimensions>::find_last_usable_accumulator(Color perspective) const noexcept {

    for (std::size_t curr_idx = size - 1; curr_idx > 0; curr_idx--)
    {
        if (acc[curr_idx].acc().computed[perspective])
            return curr_idx;

        if (PSQFeatureSet::requires_refresh(acc[curr_idx].diff, perspective))
            return curr_idx;
    }

    return 0;
}

template<IndexType Dimensions>
std::size_t
UpdateThreats<Dimensions>::find_last_usable_accumulator(Color perspective) const noexcept {

    for (std::size_t curr_idx = size - 1; curr_idx > 0; curr_idx--)
    {
        if (acc[curr_idx].acc().computed[perspective])
            return curr_idx;

        if (ThreatFeatureSet::requires_refresh(acc[curr_idx].diff, perspective))
            return curr_idx;
    }

    return 0;
}

template<IndexType Dimensions>
void UpdateHalfka<Dimensions>::forward_update_incremental(
  Color                                 perspective,
  const Position&                       pos,
  const FeatureTransformer<Dimensions>& featureTransformer,
  const std::size_t                     begin) noexcept {

    assert(begin < acc.size());
    assert(acc[begin].acc().computed[perspective]);

    const Square ksq = pos.square<KING>(perspective);

    for (std::size_t next = begin + 1; next < size; next++)
    {
        if (next + 1 < size)
        {
            DirtyPiece& dp1 = acc[next].diff;
            DirtyPiece& dp2 = acc[next + 1].diff;

            if (dp1.to != SQ_NONE && dp1.to == dp2.remove_sq)
            {
                const Square captureSq = dp1.to;
                dp1.to = dp2.remove_sq = SQ_NONE;
                double_inc_update(perspective, featureTransformer, ksq, acc[next], acc[next + 1],
                                  acc[next - 1]);
                dp1.to = dp2.remove_sq = captureSq;
                next++;
                continue;
            }
        }

        update_accumulator_incremental<true, PSQFeatureSet>(perspective, featureTransformer, ksq,
                                                            acc[next], acc[next - 1]);
    }

    assert(acc[size - 1].acc().computed[perspective]);
}

template<IndexType Dimensions>
void UpdateThreats<Dimensions>::forward_update_incremental(
  Color                                                                         perspective,
  const Position&                                                               pos,
  const FeatureTransformer<Dimensions>&                                         featureTransformer,
  const std::array<AccumulatorStateSimple<PSQFeatureSet, Dimensions>, MaxSize>& psqtAcc,
  const std::size_t                                                             begin) noexcept {

    assert(begin < acc.size());
    assert(acc[begin].acc().computed[perspective]);

    const Square ksq = pos.square<KING>(perspective);

    for (std::size_t next = begin + 1; next < size; next++)
    {
        if (next + 1 < size)
        {
            const DirtyPiece& dp2 = psqtAcc[next + 1].diff;

            if (dp2.remove_sq != SQ_NONE
                && (acc[next].diff.threateningSqs & square_bb(dp2.remove_sq)))
            {
                double_inc_update(perspective, featureTransformer, ksq, acc[next], acc[next + 1],
                                  acc[next - 1], dp2);
                next++;
                continue;
            }
        }

        update_accumulator_incremental<true, ThreatFeatureSet>(perspective, featureTransformer, ksq,
                                                               acc[next], acc[next - 1]);
    }

    assert(acc[size - 1].acc().computed[perspective]);
}

template<IndexType Dimensions>
void UpdateHalfka<Dimensions>::backward_update_incremental(
  Color                                 perspective,
  const Position&                       pos,
  const FeatureTransformer<Dimensions>& featureTransformer,
  const std::size_t                     end) noexcept {

    assert(end < acc.size());
    assert(end < size);
    assert(latest().acc().computed[perspective]);

    const Square ksq = pos.square<KING>(perspective);

    for (std::int64_t next = std::int64_t(size) - 2; next >= std::int64_t(end); next--)
        update_accumulator_incremental<false, PSQFeatureSet>(perspective, featureTransformer, ksq,
                                                             acc[next], acc[next + 1]);

    assert(acc[end].acc().computed[perspective]);
}

template<IndexType Dimensions>
void UpdateThreats<Dimensions>::backward_update_incremental(
  Color                                 perspective,
  const Position&                       pos,
  const FeatureTransformer<Dimensions>& featureTransformer,
  const std::size_t                     end) noexcept {

    assert(end < acc.size());
    assert(end < size);
    assert(latest().acc().computed[perspective]);

    const Square ksq = pos.square<KING>(perspective);

    for (std::int64_t next = std::int64_t(size) - 2; next >= std::int64_t(end); next--)
        update_accumulator_incremental<false, ThreatFeatureSet>(perspective, featureTransformer,
                                                                ksq, acc[next], acc[next + 1]);

    assert(acc[end].acc().computed[perspective]);
}

// Explicit template instantiations for the currently supported dimensions.
template struct UpdateHalfka<TransformedFeatureDimensionsBig>;
template struct UpdateHalfka<TransformedFeatureDimensionsSmall>;
template struct UpdateThreats<TransformedFeatureDimensionsBig>;

namespace {

Bitboard get_changed_pieces(const std::array<Piece, SQUARE_NB>& oldPieces,
                            const std::array<Piece, SQUARE_NB>& newPieces) {
#if defined(USE_AVX512) || defined(USE_AVX2)
    static_assert(sizeof(Piece) == 1);
    Bitboard sameBB = 0;

    for (int i = 0; i < 64; i += 32)
    {
        const __m256i old_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&oldPieces[i]));
        const __m256i new_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&newPieces[i]));
        const __m256i cmpEqual        = _mm256_cmpeq_epi8(old_v, new_v);
        const std::uint32_t equalMask = _mm256_movemask_epi8(cmpEqual);
        sameBB |= static_cast<Bitboard>(equalMask) << i;
    }
    return ~sameBB;
#elif defined(USE_NEON)
    uint8x16x4_t old_v = vld4q_u8(reinterpret_cast<const uint8_t*>(oldPieces.data()));
    uint8x16x4_t new_v = vld4q_u8(reinterpret_cast<const uint8_t*>(newPieces.data()));
    auto         cmp   = [=](const int i) { return vceqq_u8(old_v.val[i], new_v.val[i]); };

    uint8x16_t cmp0_1 = vsriq_n_u8(cmp(1), cmp(0), 1);
    uint8x16_t cmp2_3 = vsriq_n_u8(cmp(3), cmp(2), 1);
    uint8x16_t merged = vsriq_n_u8(cmp2_3, cmp0_1, 2);
    merged            = vsriq_n_u8(merged, merged, 4);
    uint8x8_t sameBB  = vshrn_n_u16(vreinterpretq_u16_u8(merged), 4);

    return ~vget_lane_u64(vreinterpret_u64_u8(sameBB), 0);
#else
    Bitboard changed = 0;

    for (Square sq = SQUARE_ZERO; sq < SQUARE_NB; ++sq)
        changed |= static_cast<Bitboard>(oldPieces[sq] != newPieces[sq]) << sq;

    return changed;
#endif
}

template<IndexType Dimensions>
void update_accumulator_refresh_cache(
  Color                                              perspective,
  const FeatureTransformer<Dimensions>&              featureTransformer,
  const Position&                                    pos,
  AccumulatorStateSimple<PSQFeatureSet, Dimensions>& accumulatorState,
  AccumulatorCaches::Cache<Dimensions>&              cache) {

    using Tiling [[maybe_unused]] = SIMDTiling<Dimensions, Dimensions, PSQTBuckets>;

    const Square             ksq   = pos.square<KING>(perspective);
    auto&                    entry = cache[ksq][perspective];
    PSQFeatureSet::IndexList removed, added;

    const Bitboard changedBB = get_changed_pieces(entry.pieces, pos.piece_array());
    Bitboard       removedBB = changedBB & entry.pieceBB;
    Bitboard       addedBB   = changedBB & pos.pieces();

    while (removedBB)
    {
        Square sq = pop_lsb(removedBB);
        removed.push_back(PSQFeatureSet::make_index(perspective, sq, entry.pieces[sq], ksq));
    }
    while (addedBB)
    {
        Square sq = pop_lsb(addedBB);
        added.push_back(PSQFeatureSet::make_index(perspective, sq, pos.piece_on(sq), ksq));
    }

    entry.pieceBB = pos.pieces();
    entry.pieces  = pos.piece_array();

    auto& accumulator                 = accumulatorState.acc();
    accumulator.computed[perspective] = true;

#ifdef VECTOR
    vec_t      accVec[Tiling::NumRegs];
    psqt_vec_t psqtVec[Tiling::NumPsqtRegs];

    const auto* weights = &featureTransformer.weights[0];

    for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
    {
        auto* accTile =
          reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][j * Tiling::TileHeight]);
        auto* entryTile = reinterpret_cast<vec_t*>(&entry.accumulation[j * Tiling::TileHeight]);

        for (IndexType k = 0; k < Tiling::NumRegs; ++k)
            accVec[k] = entryTile[k];

        int i = 0;
        for (; i < std::min(removed.ssize(), added.ssize()); ++i)
        {
            size_t       indexR  = removed[i];
            const size_t offsetR = Dimensions * indexR;
            auto*        columnR = reinterpret_cast<const vec_t*>(&weights[offsetR]);
            size_t       indexA  = added[i];
            const size_t offsetA = Dimensions * indexA;
            auto*        columnA = reinterpret_cast<const vec_t*>(&weights[offsetA]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                accVec[k] = fused<Vec16Wrapper, Add, Sub>(accVec[k], columnA[k], columnR[k]);
        }
        for (; i < removed.ssize(); ++i)
        {
            size_t       index  = removed[i];
            const size_t offset = Dimensions * index;
            auto*        column = reinterpret_cast<const vec_t*>(&weights[offset]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                accVec[k] = vec_sub_16(accVec[k], column[k]);
        }
        for (; i < added.ssize(); ++i)
        {
            size_t       index  = added[i];
            const size_t offset = Dimensions * index;
            auto*        column = reinterpret_cast<const vec_t*>(&weights[offset]);

            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                accVec[k] = vec_add_16(accVec[k], column[k]);
        }

        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&entryTile[k], accVec[k]);
        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&accTile[k], accVec[k]);

        weights += Tiling::TileHeight;
    }

    for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
    {
        auto* accTilePsqt = reinterpret_cast<psqt_vec_t*>(
          &accumulator.psqtAccumulation[perspective][j * Tiling::PsqtTileHeight]);
        auto* entryTilePsqt =
          reinterpret_cast<psqt_vec_t*>(&entry.psqtAccumulation[j * Tiling::PsqtTileHeight]);

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            psqtVec[k] = entryTilePsqt[k];

        for (int i = 0; i < removed.ssize(); ++i)
        {
            size_t       index  = removed[i];
            const size_t offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*        columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqtVec[k] = vec_sub_psqt_32(psqtVec[k], columnPsqt[k]);
        }
        for (int i = 0; i < added.ssize(); ++i)
        {
            size_t       index  = added[i];
            const size_t offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*        columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.psqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqtVec[k] = vec_add_psqt_32(psqtVec[k], columnPsqt[k]);
        }

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&entryTilePsqt[k], psqtVec[k]);
        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&accTilePsqt[k], psqtVec[k]);
    }

#else

    for (const auto index : removed)
    {
        const IndexType offset = Dimensions * index;
        for (IndexType j = 0; j < Dimensions; ++j)
            entry.accumulation[j] -= featureTransformer.weights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            entry.psqtAccumulation[k] -= featureTransformer.psqtWeights[index * PSQTBuckets + k];
    }
    for (const auto index : added)
    {
        const IndexType offset = Dimensions * index;
        for (IndexType j = 0; j < Dimensions; ++j)
            entry.accumulation[j] += featureTransformer.weights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            entry.psqtAccumulation[k] += featureTransformer.psqtWeights[index * PSQTBuckets + k];
    }

    accumulator.accumulation[perspective]     = entry.accumulation;
    accumulator.psqtAccumulation[perspective] = entry.psqtAccumulation;
#endif
}

template<IndexType Dimensions>
void update_threats_accumulator_full(
  Color                                                 perspective,
  const FeatureTransformer<Dimensions>&                 featureTransformer,
  const Position&                                       pos,
  AccumulatorStateSimple<ThreatFeatureSet, Dimensions>& accumulatorState) {
    using Tiling [[maybe_unused]] = SIMDTiling<Dimensions, Dimensions, PSQTBuckets>;

    ThreatFeatureSet::IndexList active;
    ThreatFeatureSet::append_active_indices(perspective, pos, active);

    auto& accumulator                 = accumulatorState.acc();
    accumulator.computed[perspective] = true;

#ifdef VECTOR
    vec_t      accVec[Tiling::NumRegs];
    psqt_vec_t psqtVec[Tiling::NumPsqtRegs];

    const auto* threatWeights = &featureTransformer.threatWeights[0];

    for (IndexType j = 0; j < Dimensions / Tiling::TileHeight; ++j)
    {
        auto* accTile =
          reinterpret_cast<vec_t*>(&accumulator.accumulation[perspective][j * Tiling::TileHeight]);

        for (IndexType k = 0; k < Tiling::NumRegs; ++k)
            accVec[k] = vec_zero();

        int i = 0;

        for (; i < active.ssize(); ++i)
        {
            size_t       index  = active[i];
            const size_t offset = Dimensions * index;
            auto*        column = reinterpret_cast<const vec_i8_t*>(&threatWeights[offset]);

    #ifdef USE_NEON
            for (IndexType k = 0; k < Tiling::NumRegs; k += 2)
            {
                accVec[k]     = vec_add_16(accVec[k], vmovl_s8(vget_low_s8(column[k / 2])));
                accVec[k + 1] = vec_add_16(accVec[k + 1], vmovl_high_s8(column[k / 2]));
            }
    #else
            for (IndexType k = 0; k < Tiling::NumRegs; ++k)
                accVec[k] = vec_add_16(accVec[k], vec_convert_8_16(column[k]));
    #endif
        }

        for (IndexType k = 0; k < Tiling::NumRegs; k++)
            vec_store(&accTile[k], accVec[k]);

        threatWeights += Tiling::TileHeight;
    }

    for (IndexType j = 0; j < PSQTBuckets / Tiling::PsqtTileHeight; ++j)
    {
        auto* accTilePsqt = reinterpret_cast<psqt_vec_t*>(
          &accumulator.psqtAccumulation[perspective][j * Tiling::PsqtTileHeight]);

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            psqtVec[k] = vec_zero_psqt();

        for (int i = 0; i < active.ssize(); ++i)
        {
            size_t       index  = active[i];
            const size_t offset = PSQTBuckets * index + j * Tiling::PsqtTileHeight;
            auto*        columnPsqt =
              reinterpret_cast<const psqt_vec_t*>(&featureTransformer.threatPsqtWeights[offset]);

            for (std::size_t k = 0; k < Tiling::NumPsqtRegs; ++k)
                psqtVec[k] = vec_add_psqt_32(psqtVec[k], columnPsqt[k]);
        }

        for (IndexType k = 0; k < Tiling::NumPsqtRegs; ++k)
            vec_store_psqt(&accTilePsqt[k], psqtVec[k]);
    }

#else

    for (IndexType j = 0; j < Dimensions; ++j)
        accumulator.accumulation[perspective][j] = 0;

    for (std::size_t k = 0; k < PSQTBuckets; ++k)
        accumulator.psqtAccumulation[perspective][k] = 0;

    for (const auto index : active)
    {
        const IndexType offset = Dimensions * index;

        for (IndexType j = 0; j < Dimensions; ++j)
            accumulator.accumulation[perspective][j] +=
              featureTransformer.threatWeights[offset + j];

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            accumulator.psqtAccumulation[perspective][k] +=
              featureTransformer.threatPsqtWeights[index * PSQTBuckets + k];
    }

#endif
}

template<IndexType Dimensions>
void evaluate_psqt_side(UpdateHalfka<Dimensions>&             halfka,
                        Color                                 perspective,
                        const Position&                       pos,
                        const FeatureTransformer<Dimensions>& featureTransformer,
                        AccumulatorCaches::Cache<Dimensions>& cache) noexcept {

    const auto last = halfka.find_last_usable_accumulator(perspective);

    if (halfka.acc[last].acc().computed[perspective])
        halfka.forward_update_incremental(perspective, pos, featureTransformer, last);
    else
    {
        update_accumulator_refresh_cache(perspective, featureTransformer, pos, halfka.latest(),
                                         cache);
        halfka.backward_update_incremental(perspective, pos, featureTransformer, last);
    }
}

template<IndexType Dimensions>
void evaluate_threats_side(UpdateThreats<Dimensions>&            threats,
                           UpdateHalfka<Dimensions>&             psqt,
                           Color                                 perspective,
                           const Position&                       pos,
                           const FeatureTransformer<Dimensions>& featureTransformer) noexcept {

    const auto last = threats.find_last_usable_accumulator(perspective);

    if (threats.acc[last].acc().computed[perspective])
        threats.forward_update_incremental(perspective, pos, featureTransformer, psqt.acc, last);
    else
    {
        update_threats_accumulator_full(perspective, featureTransformer, pos, threats.latest());
        threats.backward_update_incremental(perspective, pos, featureTransformer, last);
    }
}

}  // namespace

void BigNetworkAccumulator::reset() noexcept {
    psqt.reset_empty();
    threat.reset_empty();
}

void BigNetworkAccumulator::push(const DirtyBoardData& dirtyBoardData) noexcept {
    psqt.reset().diff   = dirtyBoardData.dp;
    threat.reset().diff = dirtyBoardData.dts;
}

void BigNetworkAccumulator::pop() noexcept {
    psqt.pop();
    threat.pop();
}

void BigNetworkAccumulator::evaluate(
  const Position&                                            pos,
  const FeatureTransformer<TransformedFeatureDimensionsBig>& featureTransformer,
  AccumulatorCaches::Cache<TransformedFeatureDimensionsBig>& cache) noexcept {

    evaluate_psqt_side(psqt, WHITE, pos, featureTransformer, cache);
    evaluate_threats_side(threat, psqt, WHITE, pos, featureTransformer);
    evaluate_psqt_side(psqt, BLACK, pos, featureTransformer, cache);
    evaluate_threats_side(threat, psqt, BLACK, pos, featureTransformer);
}

void SmallNetworkAccumulator::reset() noexcept { psqt.reset_empty(); }

void SmallNetworkAccumulator::push(const DirtyPiece& dp) noexcept { psqt.reset().diff = dp; }

void SmallNetworkAccumulator::pop() noexcept { psqt.pop(); }

void SmallNetworkAccumulator::evaluate(
  const Position&                                              pos,
  const FeatureTransformer<TransformedFeatureDimensionsSmall>& featureTransformer,
  AccumulatorCaches::Cache<TransformedFeatureDimensionsSmall>& cache) noexcept {

    evaluate_psqt_side(psqt, WHITE, pos, featureTransformer, cache);
    evaluate_psqt_side(psqt, BLACK, pos, featureTransformer, cache);
}

}  // namespace Stockfish::Eval::NNUE
