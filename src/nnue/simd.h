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

#ifndef NNUE_SIMD_H_INCLUDED
#define NNUE_SIMD_H_INCLUDED

#if defined(USE_AVX2)
    #include <immintrin.h>

#elif defined(USE_SSE41)
    #include <smmintrin.h>

#elif defined(USE_SSSE3)
    #include <tmmintrin.h>

#elif defined(USE_SSE2)
    #include <emmintrin.h>

#elif defined(USE_NEON)
    #include <arm_neon.h>
#endif

#include "../types.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE::SIMD {

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
#define VECTOR

#ifdef USE_AVX512
using vec_t      = __m512i;
using vec_i8_t   = __m256i;
using vec128_t   = __m128i;
using psqt_vec_t = __m256i;
using vec_uint_t = __m512i;

constexpr int NumRegistersSIMD = 16;
constexpr int MaxChunkSize     = 64;

inline vec_t vec_load(const vec_t* a) { return _mm512_load_si512(a); }
inline void  vec_store(vec_t* a, vec_t b) { _mm512_store_si512(a, b); }
inline vec_t vec_convert8to16(vec_i8_t a) { return _mm512_cvtepi8_epi16(a); }
inline vec_t vec_add16(vec_t a, vec_t b) { return _mm512_add_epi16(a, b); }
inline vec_t vec_sub16(vec_t a, vec_t b) { return _mm512_sub_epi16(a, b); }
inline vec_t vec_mulhi16(vec_t a, vec_t b) { return _mm512_mulhi_epi16(a, b); }
inline vec_t vec_zero16() { return _mm512_setzero_epi32(); }
inline vec_t vec_set16(int a) { return _mm512_set1_epi16(a); }
inline vec_t vec_max16(vec_t a, vec_t b) { return _mm512_max_epi16(a, b); }
inline vec_t vec_min16(vec_t a, vec_t b) { return _mm512_min_epi16(a, b); }
inline vec_t vec_slli16(vec_t a, int b) { return _mm512_slli_epi16(a, b); }
inline vec_t vec_set32(int a) { return _mm512_set1_epi32(a); }
inline vec_t vec_add32(vec_t a, vec_t b) { return _mm512_add_epi32(a, b); }
// Inverse permuted at load time
inline vec_t vec_packus16(vec_t a, vec_t b) { return _mm512_packus_epi16(a, b); }

inline psqt_vec_t vec_load_psqt(const psqt_vec_t* a) { return _mm256_load_si256(a); }
inline void       vec_store_psqt(psqt_vec_t* a, psqt_vec_t b) { _mm256_store_si256(a, b); }
inline psqt_vec_t vec_add32(psqt_vec_t a, psqt_vec_t b) { return _mm256_add_epi32(a, b); }
inline psqt_vec_t vec_sub32(psqt_vec_t a, psqt_vec_t b) { return _mm256_sub_epi32(a, b); }
inline psqt_vec_t vec_zero32() { return _mm256_setzero_si256(); }

inline vec128_t vec128_zero16() { return _mm_setzero_si128(); }
inline vec128_t vec128_set16(int a) { return _mm_set1_epi16(a); }
inline vec128_t vec128_load(const vec128_t* a) { return _mm_load_si128(a); }
inline void     vec128_storeu(vec128_t* a, vec128_t b) { _mm_storeu_si128(a, b); }
inline vec128_t vec128_add16(vec128_t a, vec128_t b) { return _mm_add_epi16(a, b); }

inline auto vec_nnz(vec_uint_t a) { return _mm512_cmpgt_epi32_mask(a, _mm512_setzero_si512()); }

#elif USE_AVX2
using vec_t      = __m256i;
using vec_i8_t   = __m128i;
using vec128_t   = __m128i;
using psqt_vec_t = __m256i;
using vec_uint_t = __m256i;

constexpr int NumRegistersSIMD = 12;
constexpr int MaxChunkSize     = 32;

inline vec_t vec_load(const vec_t* a) { return _mm256_load_si256(a); }
inline void  vec_store(vec_t* a, vec_t b) { _mm256_store_si256(a, b); }
inline vec_t vec_convert8to16(vec_i8_t a) { return _mm256_cvtepi8_epi16(a); }
inline vec_t vec_add16(vec_t a, vec_t b) { return _mm256_add_epi16(a, b); }
inline vec_t vec_sub16(vec_t a, vec_t b) { return _mm256_sub_epi16(a, b); }
inline vec_t vec_mulhi16(vec_t a, vec_t b) { return _mm256_mulhi_epi16(a, b); }
inline vec_t vec_zero16() { return _mm256_setzero_si256(); }
inline vec_t vec_set16(int a) { return _mm256_set1_epi16(a); }
inline vec_t vec_max16(vec_t a, vec_t b) { return _mm256_max_epi16(a, b); }
inline vec_t vec_min16(vec_t a, vec_t b) { return _mm256_min_epi16(a, b); }
inline vec_t vec_slli16(vec_t a, int b) { return _mm256_slli_epi16(a, b); }
inline vec_t vec_set32(int a) { return _mm256_set1_epi32(a); }
inline vec_t vec_add32(vec_t a, vec_t b) { return _mm256_add_epi32(a, b); }
// Inverse permuted at load time
inline vec_t vec_packus16(vec_t a, vec_t b) { return _mm256_packus_epi16(a, b); }

inline psqt_vec_t vec_load_psqt(const psqt_vec_t* a) { return _mm256_load_si256(a); }
inline void       vec_store_psqt(psqt_vec_t* a, psqt_vec_t b) { _mm256_store_si256(a, b); }
inline psqt_vec_t vec_add32(psqt_vec_t a, psqt_vec_t b) { return _mm256_add_epi32(a, b); }
inline psqt_vec_t vec_sub32(psqt_vec_t a, psqt_vec_t b) { return _mm256_sub_epi32(a, b); }
inline psqt_vec_t vec_zero32() { return _mm256_setzero_si256(); }

inline vec128_t vec128_zero16() { return _mm_setzero_si128(); }
inline vec128_t vec128_set16(int a) { return _mm_set1_epi16(a); }
inline vec128_t vec128_load(const vec128_t* a) { return _mm_load_si128(a); }
inline void     vec128_storeu(vec128_t* a, vec128_t b) { _mm_storeu_si128(a, b); }
inline vec128_t vec128_add16(vec128_t a, vec128_t b) { return _mm_add_epi16(a, b); }

inline auto vec_nnz(vec_uint_t a) {
    #if defined(USE_VNNI) && !defined(USE_AVXVNNI)
    return _mm256_cmpgt_epi32_mask(a, _mm256_setzero_si256());
    #else
    return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a, _mm256_setzero_si256())));
    #endif
}

#elif USE_SSE2
using vec_t      = __m128i;
using vec_i8_t   = std::uint64_t;  // for the correct size -- will be loaded into an xmm reg
using vec128_t   = __m128i;
using psqt_vec_t = __m128i;
using vec_uint_t = __m128i;
constexpr int NumRegistersSIMD = (Is64Bit ? 12 : 6);
constexpr int MaxChunkSize     = 16;

inline vec_t vec_load(const vec_t* a) { return *a; }
inline void  vec_store(vec_t* a, vec_t b) { *a = b; }
inline vec_t vec_add16(vec_t a, vec_t b) { return _mm_add_epi16(a, b); }
inline vec_t vec_sub16(vec_t a, vec_t b) { return _mm_sub_epi16(a, b); }
inline vec_t vec_mulhi16(vec_t a, vec_t b) { return _mm_mulhi_epi16(a, b); }
inline vec_t vec_zero16() { return _mm_setzero_si128(); }
inline vec_t vec_set16(int a) { return _mm_set1_epi16(a); }
inline vec_t vec_max16(vec_t a, vec_t b) { return _mm_max_epi16(a, b); }
inline vec_t vec_min16(vec_t a, vec_t b) { return _mm_min_epi16(a, b); }
inline vec_t vec_slli16(vec_t a, int b) { return _mm_slli_epi16(a, b); }
inline vec_t vec_set32(int a) { return _mm_set1_epi32(a); }
inline vec_t vec_add32(vec_t a, vec_t b) { return _mm_add_epi32(a, b); }
inline vec_t vec_packus16(vec_t a, vec_t b) { return _mm_packus_epi16(a, b); }

inline psqt_vec_t vec_load_psqt(const psqt_vec_t* a) { return *a; }
inline void       vec_store_psqt(psqt_vec_t* a, psqt_vec_t b) { *a = b; }
inline psqt_vec_t vec_add32(psqt_vec_t a, psqt_vec_t b) { return _mm_add_epi32(a, b); }
inline psqt_vec_t vec_sub32(psqt_vec_t a, psqt_vec_t b) { return _mm_sub_epi32(a, b); }
inline psqt_vec_t vec_zero32() { return _mm_setzero_si128(); }

inline unsigned vec_nnz(vec_uint_t a) {
    return static_cast<unsigned>(
      _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a, _mm_setzero_si128()))));
}

    #ifdef __i386__
inline __m128i _mm_cvtsi64_si128(int64_t val) {
    return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&val));
}
    #endif

#ifdef USE_SSE41
inline vec_t vec_convert8to16(vec_i8_t a) {
    return _mm_cvtepi8_epi16(_mm_cvtsi64_si128(static_cast<int64_t>(a)));
}
#else
// Credit: Yoshie2000
inline __m128i vec_convert8to16(uint64_t x) {
    __m128i v8   = _mm_cvtsi64_si128(static_cast<int64_t>(x));
    __m128i sign = _mm_cmpgt_epi8(_mm_setzero_si128(), v8);
    return _mm_unpacklo_epi8(v8, sign);
}
#endif

inline vec128_t vec128_zero16() { return _mm_setzero_si128(); }
inline vec128_t vec128_set16(int a) { return _mm_set1_epi16(a); }
inline vec128_t vec128_load(const vec128_t* a) { return _mm_load_si128(a); }
inline void     vec128_storeu(vec128_t* a, vec128_t b) { _mm_storeu_si128(a, b); }
inline vec128_t vec128_add16(vec128_t a, vec128_t b) { return _mm_add_epi16(a, b); }

#elif USE_NEON
using vec_t      = int16x8_t;
using vec_i8_t   = int8x16_t;
using psqt_vec_t = int32x4_t;
using vec128_t   = uint16x8_t;
using vec_uint_t = uint32x4_t;

constexpr int NumRegistersSIMD = 16;
constexpr int MaxChunkSize     = 16;

inline vec_t vec_load(const vec_t* a) { return *a; }
inline void  vec_store(vec_t* a, vec_t b) { *a = b; }
inline vec_t vec_add16(vec_t a, vec_t b) { return vaddq_s16(a, b); }
inline vec_t vec_sub16(vec_t a, vec_t b) { return vsubq_s16(a, b); }
inline vec_t vec_mulhi16(vec_t a, vec_t b) { return vqdmulhq_s16(a, b); }
inline vec_t vec_zero16() { return vdupq_n_s16(0); }
inline vec_t vec_set16(int a) { return vdupq_n_s16(a); }
inline vec_t vec_max16(vec_t a, vec_t b) { return vmaxq_s16(a, b); }
inline vec_t vec_min16(vec_t a, vec_t b) { return vminq_s16(a, b); }
inline vec_t vec_slli16(vec_t a, int b) { return vshlq_s16(a, vec_set16(b)); }
inline vec_t vec_packus16(vec_t a, vec_t b) {
    return reinterpret_cast<vec_t>(vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)));
}

inline psqt_vec_t vec_load_psqt(const psqt_vec_t* a) { return *a; }
inline void       vec_store_psqt(psqt_vec_t* a, psqt_vec_t b) { *a = b; }
inline psqt_vec_t vec_add32(psqt_vec_t a, psqt_vec_t b) { return vaddq_s32(a, b); }
inline psqt_vec_t vec_sub32(psqt_vec_t a, psqt_vec_t b) { return vsubq_s32(a, b); }
inline psqt_vec_t vec_zero32() { return vdupq_n_s32(0); }

static constexpr std::uint32_t Mask[4] = {1, 2, 4, 8};
inline std::uint32_t           vec_nnz(vec_uint_t a) {
    return vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(Mask)));
}

inline vec128_t vec128_zero16() { return vdupq_n_u16(0); }
inline vec128_t vec128_set16(int a) { return vdupq_n_u16(a); }
inline vec128_t vec128_load(const vec128_t* a) {
    return vld1q_u16(reinterpret_cast<const std::uint16_t*>(a));
}
inline void vec128_storeu(vec128_t* a, vec128_t b) {
    vst1q_u16(reinterpret_cast<std::uint16_t*>(a), b);
}
inline vec128_t vec128_add16(vec128_t a, vec128_t b) { return vaddq_u16(a, b); }

    #ifndef __aarch64__
// Single instruction doesn't exist on 32-bit ARM
inline int8x16_t vmovl_high_s8(int8x16_t val) { return vmovl_s8(vget_high_s8(val)); }
    #endif

#else
    #undef VECTOR

#endif

struct Vec16Wrapper {
#ifdef VECTOR
    using type = vec_t;
    static type add(const type& lhs, const type& rhs) { return vec_add16(lhs, rhs); }
    static type sub(const type& lhs, const type& rhs) { return vec_sub16(lhs, rhs); }
#else
    using type = BiasType;
    static type add(const type& lhs, const type& rhs) { return lhs + rhs; }
    static type sub(const type& lhs, const type& rhs) { return lhs - rhs; }
#endif
};

struct Vec32Wrapper {
#ifdef VECTOR
    using type = psqt_vec_t;
    static type add(const type& lhs, const type& rhs) { return vec_add32(lhs, rhs); }
    static type sub(const type& lhs, const type& rhs) { return vec_sub32(lhs, rhs); }
#else
    using type = PSQTWeightType;
    static type add(const type& lhs, const type& rhs) { return lhs + rhs; }
    static type sub(const type& lhs, const type& rhs) { return lhs - rhs; }
#endif
};

enum UpdateOperation {
    Add,
    Sub
};

template<typename VecWrapper,
         UpdateOperation... ops,
         std::enable_if_t<sizeof...(ops) == 0, bool> = true>
typename VecWrapper::type fused(const typename VecWrapper::type& in) {
    return in;
}

template<typename VecWrapper,
         UpdateOperation update_op,
         UpdateOperation... ops,
         typename T,
         typename... Ts,
         std::enable_if_t<is_all_same_v<typename VecWrapper::type, T, Ts...>, bool> = true,
         std::enable_if_t<sizeof...(ops) == sizeof...(Ts), bool>                    = true>
typename VecWrapper::type
fused(const typename VecWrapper::type& in, const T& operand, const Ts&... operands) {
    switch (update_op)
    {
    case Add :
        return fused<VecWrapper, ops...>(VecWrapper::add(in, operand), operands...);
    case Sub :
        return fused<VecWrapper, ops...>(VecWrapper::sub(in, operand), operands...);
    default :
        static_assert(update_op == Add || update_op == Sub,
                      "Only Add and Sub are currently supported.");
        return typename VecWrapper::type();
    }
}

#if defined(USE_AVX512)

[[maybe_unused]] static int m512_hadd(__m512i sum, int bias) {
    return _mm512_reduce_add_epi32(sum) + bias;
}

[[maybe_unused]] static void m512_add_dpbusd_epi32(__m512i& acc, __m512i a, __m512i b) {

    #if defined(USE_VNNI)
    acc = _mm512_dpbusd_epi32(acc, a, b);
    #else
    __m512i product0 = _mm512_maddubs_epi16(a, b);
    product0         = _mm512_madd_epi16(product0, _mm512_set1_epi16(1));
    acc              = _mm512_add_epi32(acc, product0);
    #endif
}

#endif

#if defined(USE_AVX2)

[[maybe_unused]] static int m256_hadd(__m256i sum, int bias) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
    sum128         = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
    return _mm_cvtsi128_si32(sum128) + bias;
}

[[maybe_unused]] static void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {

    #if defined(USE_VNNI)
    acc = _mm256_dpbusd_epi32(acc, a, b);
    #else
    __m256i product0 = _mm256_maddubs_epi16(a, b);
    product0         = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    acc              = _mm256_add_epi32(acc, product0);
    #endif
}

#endif

#if defined(USE_SSSE3)

[[maybe_unused]] static int m128_hadd(__m128i sum, int bias) {
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E));  //_MM_PERM_BADC
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1));  //_MM_PERM_CDAB
    return _mm_cvtsi128_si32(sum) + bias;
}

[[maybe_unused]] static void m128_add_dpbusd_epi32(__m128i& acc, __m128i a, __m128i b) {

    __m128i product0 = _mm_maddubs_epi16(a, b);
    product0         = _mm_madd_epi16(product0, _mm_set1_epi16(1));
    acc              = _mm_add_epi32(acc, product0);
}

#endif

#if defined(USE_NEON_DOTPROD)

[[maybe_unused]] static void
dotprod_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {

    acc = vdotq_s32(acc, a, b);
}
#endif

#if defined(USE_NEON)

[[maybe_unused]] static int neon_m128_reduce_add_epi32(int32x4_t s) {
    #if USE_NEON >= 8
    return vaddvq_s32(s);
    #else
    return s[0] + s[1] + s[2] + s[3];
    #endif
}

[[maybe_unused]] static int neon_m128_hadd(int32x4_t sum, int bias) {
    return neon_m128_reduce_add_epi32(sum) + bias;
}

#endif

#if USE_NEON >= 8
[[maybe_unused]] static void neon_m128_add_dpbusd_epi32(int32x4_t& acc, int8x16_t a, int8x16_t b) {

    int16x8_t product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t product1 = vmull_high_s8(a, b);
    int16x8_t sum      = vpaddq_s16(product0, product1);
    acc                = vpadalq_s16(acc, sum);
}
#endif


struct DotProduct {
#if defined(USE_AVX512)
    using input_vec = __m512i;
    using accum_vec = __m512i;
    static constexpr int simd_width = sizeof(accum_vec) / sizeof(std::int32_t);

    static input_vec splat(int value) { return _mm512_set1_epi32(value); }

    static accum_vec zero() { return _mm512_setzero_si512(); }

    static void madd(accum_vec& acc, input_vec a, const input_vec& b) {
        m512_add_dpbusd_epi32(acc, a, b);
    }

    static accum_vec add(accum_vec a, accum_vec b) { return _mm512_add_epi32(a, b); }

    static int horizontal_add(accum_vec sum, int bias) { return m512_hadd(sum, bias); }
#elif defined(USE_AVX2)
    using input_vec = __m256i;
    using accum_vec = __m256i;
    static constexpr int simd_width = sizeof(accum_vec) / sizeof(std::int32_t);

    static input_vec splat(int value) { return _mm256_set1_epi32(value); }

    static accum_vec zero() { return _mm256_setzero_si256(); }

    static void madd(accum_vec& acc, input_vec a, const input_vec& b) {
        m256_add_dpbusd_epi32(acc, a, b);
    }

    static accum_vec add(accum_vec a, accum_vec b) { return _mm256_add_epi32(a, b); }

    static int horizontal_add(accum_vec sum, int bias) { return m256_hadd(sum, bias); }

#elif defined(USE_SSSE3)
    using input_vec = __m128i;
    using accum_vec = __m128i;
    static constexpr int simd_width = sizeof(accum_vec) / sizeof(std::int32_t);

    static input_vec splat(int value) { return _mm_set1_epi32(value); }

    static accum_vec zero() { return _mm_setzero_si128(); }

    static void madd(accum_vec& acc, input_vec a, const input_vec& b) {
        m128_add_dpbusd_epi32(acc, a, b);
    }

    static accum_vec add(accum_vec a, accum_vec b) { return _mm_add_epi32(a, b); }

    static int horizontal_add(accum_vec sum, int bias) { return m128_hadd(sum, bias); }

#elif defined(USE_NEON_DOTPROD) || USE_NEON >= 8
    using input_vec = int8x16_t;
    using accum_vec = int32x4_t;
    static constexpr int simd_width = sizeof(accum_vec) / sizeof(std::int32_t);

    static input_vec splat(int value) {
        return vreinterpretq_s8_u32(vdupq_n_u32(static_cast<std::uint32_t>(value)));
    }

    static accum_vec zero() { return vdupq_n_s32(0); }

    static void madd(accum_vec& acc, input_vec a, const input_vec& b) {
    #if defined(USE_NEON_DOTPROD)
        dotprod_m128_add_dpbusd_epi32(acc, a, b);
    #else
        neon_m128_add_dpbusd_epi32(acc, a, b);
    #endif
    }

    static accum_vec add(accum_vec a, accum_vec b) { return vaddq_s32(a, b); }

    static int horizontal_add(accum_vec sum, int bias) { return neon_m128_hadd(sum, bias); }
#endif
};

// Compute optimal SIMD register count for feature transformer accumulation.
template<IndexType TransformedFeatureWidth, IndexType HalfDimensions, IndexType PSQTBuckets>
class SIMDTiling {
#ifdef VECTOR
        // We use __m* types as template arguments, which causes GCC to emit warnings
        // about losing some attribute information. This is irrelevant to us as we
        // only take their size, so the following pragma are harmless.
    #if defined(__GNUC__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif

    template<typename SIMDRegisterType, typename LaneType, int NumLanes, int MaxRegisters>
    static constexpr int BestRegisterCount() {
        constexpr std::size_t RegisterSize = sizeof(SIMDRegisterType);
        constexpr std::size_t LaneSize     = sizeof(LaneType);

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

   public:
    static constexpr int NumRegs =
      BestRegisterCount<vec_t, WeightType, TransformedFeatureWidth, NumRegistersSIMD>();
    static constexpr int NumPsqtRegs =
      BestRegisterCount<psqt_vec_t, PSQTWeightType, PSQTBuckets, NumRegistersSIMD>();

    static constexpr IndexType TileHeight     = NumRegs * sizeof(vec_t) / 2;
    static constexpr IndexType PsqtTileHeight = NumPsqtRegs * sizeof(psqt_vec_t) / 4;

    static_assert(HalfDimensions % TileHeight == 0, "TileHeight must divide HalfDimensions");
    static_assert(PSQTBuckets % PsqtTileHeight == 0, "PsqtTileHeight must divide PSQTBuckets");
#endif
};
}

#endif
