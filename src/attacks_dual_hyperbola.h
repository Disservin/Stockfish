#ifndef ATTACKS_DUAL_HYPERBOLA_H_INCLUDED
#define ATTACKS_DUAL_HYPERBOLA_H_INCLUDED

#include <cstdint>
#include <utility>

#include "bitboard.h"

namespace Stockfish::Attacks {

struct DualMagic {
    // file, diagonal, unused, antidiagonal
    Bitboard maskFile, maskDiag, maskNone, maskAntidiag;
    // Precomputed 2 * square_bb(sq), 2 * reverse(square_bb(sq))
    Bitboard r, rr;

    const uint8_t* RESTRICT rankAttacksLookup;
    // 8 * rank_of(sq)
    int shift;
};

void init_impl();
std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied);
Bitboard bishop_attacks_bb(Square s, Bitboard occupied);
Bitboard rook_attacks_bb(Square s, Bitboard occupied);

}  // namespace Stockfish::Attacks

#endif  // ATTACKS_DUAL_HYPERBOLA_H_INCLUDED
