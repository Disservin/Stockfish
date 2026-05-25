#ifndef ATTACKS_DUAL_HYPERBOLA_H_INCLUDED
#define ATTACKS_DUAL_HYPERBOLA_H_INCLUDED

struct DualMagic {
    // file, diagonal, unused, antidiagonal
    Bitboard maskFile, maskDiag, maskNone, maskAntidiag;
    // Precomputed 2 * square_bb(sq), 2 * reverse(square_bb(sq))
    Bitboard r, rr;

    const uint8_t* RESTRICT rankAttacksLookup;
    // 8 * rank_of(sq)
    int shift;

    // We always compute [bishop, rook] attacks at once, then rely on
    // compiler's DCE and CSE to eliminate unneeded re-computations or extractions.
    //
    // When using hyperbola quintessence, file, diagonal and antidiagonal attacks
    // can use a byte reversal rather than a full bit reversal (because all squares
    // reside in different bytes). Rank atttacks cannot. Thus, for rank attacks
    // only, we use a compact lookup table indexed by the 8 bits of the rank's occupancy.
    std::pair<Bitboard, Bitboard> both_attacks_bb(Bitboard occupied) const {
        const auto bswap = [](__m256i v) {
            return _mm256_shuffle_epi8(v, _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                          13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10, 11, 12, 13, 14, 15));
        };

        // Each lane contains a mask and we follow the same HQ algorithm as
        // given above in the ARM64 code path.
        const __m256i mask = _mm256_load_si256(reinterpret_cast<const __m256i*>(this));
        const __m256i rs   = _mm256_set1_epi64x(r);
        const __m256i rrs  = _mm256_set1_epi64x(rr);

        __m256i o      = _mm256_and_si256(mask, _mm256_set1_epi64x(occupied));
        __m256i fwd    = _mm256_sub_epi64(o, rs);
        __m256i rev    = bswap(_mm256_sub_epi64(bswap(o), rrs));
        __m256i result = _mm256_and_si256(_mm256_xor_si256(fwd, rev), mask);

        // Lane 0: rook attacks (file only); lane 1: bishop attacks.
        __m128i rookBishop =
          _mm_or_si128(_mm256_extracti128_si256(result, 1), _mm256_castsi256_si128(result));

        Bitboard rowOccupancy = rankAttacksLookup[(occupied >> shift) & 0xff];
        Bitboard rankAttacks  = rowOccupancy << shift;

        // [bishop, rook]
        return {_mm_extract_epi64(rookBishop, 1), _mm_cvtsi128_si64(rookBishop) + rankAttacks};
    }
};

const DualMagic& dual_magic(Square s);

inline std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied) {
    return dual_magic(s).both_attacks_bb(occupied);
}

inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    return both_attacks_bb(s, occupied).first;
}

inline Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    return both_attacks_bb(s, occupied).second;
}

#endif  // ATTACKS_DUAL_HYPERBOLA_H_INCLUDED
