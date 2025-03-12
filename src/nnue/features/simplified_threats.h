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

//Definition of input features Simplified_Threats of NNUE evaluation function

#ifndef NNUE_FEATURES_SIMPLIFIED_THREATS_INCLUDED
#define NNUE_FEATURES_SIMPLIFIED_THREATS_INCLUDED

#include <cstdint>
#include <vector>

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish {
struct StateInfo;
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

// Feature Simplified_Threats: Combination of the position of pieces and all
// active piece-to-piece attacks. Position mirrored such that king is always on abcd files.
class Simplified_Threats {

    // Unique number for each piece type on each square
    enum {
        PS_NONE     = 0,
        PS_W_PAWN   = 0,
        PS_W_KNIGHT = 1 * SQUARE_NB,
        PS_W_BISHOP = 2 * SQUARE_NB,
        PS_W_ROOK   = 3 * SQUARE_NB,
        PS_W_QUEEN  = 4 * SQUARE_NB,
        PS_W_KING   = 5 * SQUARE_NB,
        PS_B_PAWN   = 6 * SQUARE_NB,
        PS_B_KNIGHT = 7 * SQUARE_NB,
        PS_B_BISHOP = 8 * SQUARE_NB,
        PS_B_ROOK   = 9 * SQUARE_NB,
        PS_B_QUEEN  = 10 * SQUARE_NB,
        PS_B_KING   = 11 * SQUARE_NB,
        PS_NB       = 12 * SQUARE_NB  
    };

    static constexpr IndexType PieceSquareIndex[PIECE_NB] = {
        // From white perspective
        // Viewed from other side, W and B are reversed
        PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_W_KING, PS_NONE,
        PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_B_KING, PS_NONE};

    
    IndexType threatoffsets[PIECE_NB][SQUARE_NB+2];
    void init_threat_offsets();
   public:
    // Feature name
    static constexpr const char* Name = "Simplified_Threats(Friend)";

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t HashValue = 0x7f234cb8u;

    // Number of feature dimensions
    // Simplified threats are indexed by (piece type)(enemy)(from)(to)
    // There are 7504 valid piece-from-to combinations (including color). Thus 7504*2+768 = 15776 inputs.
    static constexpr IndexType Dimensions = 15776;

    // clang-format off
    // Orient a square according to perspective (rotates by 180 for black)
    static constexpr int OrientTBL[COLOR_NB][SQUARE_NB + 1] = {
      { SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_NONE},
      { SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8,
        SQ_A8, SQ_A8, SQ_A8, SQ_A8, SQ_H8, SQ_H8, SQ_H8, SQ_H8, SQ_NONE}
    };
    // clang-format on

    // Maximum number of simultaneously active features.
    static constexpr IndexType MaxActiveDimensions = 128;
    using IndexList                                = ValueList<IndexType, MaxActiveDimensions>;

    //Use this for storing intermediate features


    Simplified_Threats() { init_threat_offsets(); };
    // Index of a feature for a given king position and another piece on some square
    template<Color Perspective>
    IndexType make_index(Piece attkr, Square from, Square to, Piece attkd, Square ksq);

    // Get a list of indices for active features
    template<Color Perspective>
    void append_active_threats(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active);

    template<Color Perspective>
    void append_active_psq(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active);

    template<Color Perspective>
    void append_active_features(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& psq, IndexList& threats);

    // Get a list of indices for recently changed features
    template<Color Perspective>
    void append_changed_indices(Square ksq, const DirtyPiece& dp, IndexList& removed, IndexList& added);
    // Returns whether the change stored in this StateInfo means
    // that a full accumulator refresh is required.
    static bool requires_refresh(const StateInfo* st, Color perspective);
};

}  // namespace Stockfish::Eval::NNUE::Features

#endif  // #ifndef NNUE_FEATURES_SIMPLIFIED_THREATS_INCLUDED
