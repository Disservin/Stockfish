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

#ifndef BITOPS_H_INCLUDED
#define BITOPS_H_INCLUDED

#include <cassert>
#include <cstdint>

#if defined(__aarch64__)
    #include <arm_acle.h>
#endif

#if defined(_WIN64) && defined(_MSC_VER)
    #include <intrin.h>  // Microsoft header for _BitScanForward64()
#endif

#if defined(USE_POPCNT) && defined(_MSC_VER)
    #include <nmmintrin.h>  // Microsoft header for _mm_popcnt_u64()
#endif

#if !defined(NO_PREFETCH) && defined(_MSC_VER)
    #include <xmmintrin.h>
#endif

#if defined(USE_PEXT)
    #include <immintrin.h>  // Header for _pext_u64() and _pdep_u64() intrinsics
#endif

namespace Stockfish {

namespace Bitops {

inline std::uint64_t bitreverse(std::uint64_t value) {
#if defined(__has_builtin)
    #if __has_builtin(__builtin_bitreverse64)
    return __builtin_bitreverse64(value);
    #endif
#endif

#if defined(__aarch64__)
    return __rbitll(value);
#else
    value = ((value >> 1) & 0x5555555555555555ULL) | ((value & 0x5555555555555555ULL) << 1);
    value = ((value >> 2) & 0x3333333333333333ULL) | ((value & 0x3333333333333333ULL) << 2);
    value = ((value >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((value & 0x0F0F0F0F0F0F0F0FULL) << 4);
    value = ((value >> 8) & 0x00FF00FF00FF00FFULL) | ((value & 0x00FF00FF00FF00FFULL) << 8);
    value = ((value >> 16) & 0x0000FFFF0000FFFFULL) | ((value & 0x0000FFFF0000FFFFULL) << 16);
    return (value >> 32) | (value << 32);
#endif
}

constexpr int constexpr_popcount(std::uint64_t v) {
    v = v - ((v >> 1) & 0x5555555555555555ULL);
    v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<int>((v * 0x0101010101010101ULL) >> 56);
}

// Counts the number of non-zero bits in an unsigned integral value.
inline int popcount(std::uint64_t value) {

#ifndef USE_POPCNT

    return constexpr_popcount(value);

#elif defined(_MSC_VER)

    return int(_mm_popcnt_u64(value));

#else  // Assumed gcc or compatible compiler

    return __builtin_popcountll(value);

#endif
}

inline constexpr int lsb_index64[64] = {
  0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61, 54, 58, 35, 52, 50, 42,
  21, 44, 38, 32, 29, 23, 17, 11, 4,  62, 46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43,
  31, 22, 10, 45, 25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};

constexpr int constexpr_lsb(std::uint64_t v) {
    assert(v != 0);

    constexpr std::uint64_t debruijn64 = 0x03F79D71B4CB0A89ULL;
    return lsb_index64[((v ^ (v - 1)) * debruijn64) >> 58];
}

// Returns the index of the least significant bit in a non-zero value.
inline int lsb(std::uint64_t value) {
    assert(value != 0);

#if defined(__GNUC__)  // GCC, Clang, ICX

    return __builtin_ctzll(value);

#elif defined(_MSC_VER)

    unsigned long idx;
    _BitScanForward64(&idx, value);
    return int(idx);
#else  // Compiler is neither GCC nor MSVC compatible
    #error "Compiler not supported."
#endif
}

// Returns the index of the most significant bit in a non-zero value.
inline int msb(std::uint64_t value) {
    assert(value != 0);

#if defined(__GNUC__)  // GCC, Clang, ICX

    return 63 - __builtin_clzll(value);

#elif defined(_MSC_VER)

    unsigned long idx;
    _BitScanReverse64(&idx, value);
    return int(idx);
#else  // Compiler is neither GCC nor MSVC compatible
    #error "Compiler not supported."
#endif
}

constexpr std::uint64_t least_significant_bit(std::uint64_t value) {
    assert(value != 0);
    return value & (std::uint64_t(0) - value);
}

// Finds and clears the least significant bit in a non-zero value.
inline int pop_lsb(std::uint64_t& value) {
    assert(value != 0);
    const int index = lsb(value);
    value &= value - 1;
    return index;
}

#if defined(USE_PEXT)
inline std::uint64_t pext(std::uint64_t value, std::uint64_t mask) {
    return _pext_u64(value, mask);
}

inline std::uint64_t pdep(std::uint64_t value, std::uint64_t mask) {
    return _pdep_u64(value, mask);
}
#endif

}  // namespace Bitops

}  // namespace Stockfish

#endif  // #ifndef BITOPS_H_INCLUDED
