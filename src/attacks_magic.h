#ifndef ATTACKS_MAGIC_H_INCLUDED
#define ATTACKS_MAGIC_H_INCLUDED

// Magic holds all magic bitboards relevant data for a single square.
struct Magic {
    Bitboard mask;
    #ifdef USE_PEXT
    uint16_t* attacks;
    Bitboard  pseudoAttacks;
    #else
    Bitboard* attacks;
    Bitboard  magic;
    unsigned  shift;
    #endif

    // Compute the attack's index using the magic bitboards approach.
    unsigned index(Bitboard occupied) const {

    #ifdef USE_PEXT
        return unsigned(pext(occupied, mask));
    #else
        if (Is64Bit)
            return unsigned(((occupied & mask) * magic) >> shift);

        unsigned lo = unsigned(occupied) & unsigned(mask);
        unsigned hi = unsigned(occupied >> 32) & unsigned(mask >> 32);
        return (lo * unsigned(magic) ^ hi * unsigned(magic >> 32)) >> shift;
    #endif
    }

    Bitboard attacks_bb(Bitboard occupied) const {
    #ifdef USE_PEXT
        return pdep(attacks[index(occupied)], pseudoAttacks);
    #else
        return attacks[index(occupied)];
    #endif
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

#endif  // ATTACKS_MAGIC_H_INCLUDED
