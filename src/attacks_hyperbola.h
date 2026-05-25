#ifndef ATTACKS_HYPERBOLA_H_INCLUDED
#define ATTACKS_HYPERBOLA_H_INCLUDED

#include <utility>

#include "bitboard.h"

namespace Stockfish::Attacks {

// Hyperbola quintessence implementation for ARM, thanks to the availability of an
// efficient bit reversal instruction.
// See https://www.chessprogramming.org/Hyperbola_Quintessence
struct Magic {
    // For rooks: file attacks, rank attacks. For bishops: diagonal/antidiagonal
    Bitboard mask1, mask2;
    // Precomputed 2 * square_bb(sq), 2 * reverse(square_bb(sq))
    Bitboard r, rr;
};

void init_impl();
const Magic& magic(Square s, PieceType pt);
Bitboard     bishop_attacks_bb(Square s, Bitboard occupied);
Bitboard     rook_attacks_bb(Square s, Bitboard occupied);
std::pair<Bitboard, Bitboard> both_attacks_bb(Square s, Bitboard occupied);

}  // namespace Stockfish::Attacks

#endif  // ATTACKS_HYPERBOLA_H_INCLUDED
