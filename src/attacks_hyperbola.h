#ifndef ATTACKS_HYPERBOLA_H_INCLUDED
#define ATTACKS_HYPERBOLA_H_INCLUDED

inline Bitboard reverse_bb(Bitboard bb) {
    #ifdef __aarch64__
    return __rbitll(bb);
    #else  // loongarch
    Bitboard out;
    asm("bitrev.d %0, %1" : "=r"(out) : "r"(bb));
    return out;
    #endif
}

// Hyperbola quintessence implementation for ARM, thanks to the availability of an
// efficient bit reversal instruction.
// See https://www.chessprogramming.org/Hyperbola_Quintessence
struct Magic {
    // For rooks: file attacks, rank attacks. For bishops: diagonal/antidiagonal
    Bitboard mask1, mask2;
    // Precomputed 2 * square_bb(sq), 2 * reverse(square_bb(sq))
    Bitboard r, rr;

    Bitboard hyperbola(Bitboard occupied, Bitboard mask) const {
        Bitboard o   = occupied & mask;
        Bitboard fwd = o - r;
        Bitboard rev = reverse_bb(o) - rr;
        return (fwd ^ reverse_bb(rev)) & mask;
    }

    Bitboard attacks_bb(Bitboard occupied) const {
        return hyperbola(occupied, mask1) | hyperbola(occupied, mask2);
    }
};

const Magic& magic(Square s, PieceType pt);

inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    return magic(s, BISHOP).attacks_bb(occupied);
}

inline Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    return magic(s, ROOK).attacks_bb(occupied);
}

inline std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied) {
    return {bishop_attacks_bb(s, occupied), rook_attacks_bb(s, occupied)};
}

#endif  // ATTACKS_HYPERBOLA_H_INCLUDED
