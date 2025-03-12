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

#include "simplified_threats.h"
#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_accumulator.h"

namespace Stockfish::Eval::NNUE::Features {
//lookup array for indexing threats
void Simplified_Threats::init_threat_offsets() {
    int pieceoffset = 0;
    PieceType piecetbl[PIECE_NB] = {NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE,
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE};
    for (int piece = 0; piece < 16; piece++) {
        if (piecetbl[piece] != NO_PIECE_TYPE) {
            threatoffsets[piece][65] = pieceoffset;
            int squareoffset = 0;
            for (int from = SQ_A1; from <= SQ_H8; from++) {
                threatoffsets[piece][from] = squareoffset;
                if (piecetbl[piece] != PAWN) {
                    Bitboard attacks = attacks_bb(piecetbl[piece], Square(from), 0ULL);
                    squareoffset += popcount(attacks);
                }
                else if (from >= SQ_A2 && from <= SQ_H7) {
                    Bitboard attacks = pawn_attacks_bb(Color(piece/8), Square(from));
                    squareoffset += popcount(attacks);
                }
            }
            threatoffsets[piece][64] = squareoffset;
            pieceoffset += 2*squareoffset;
        }
    }
}

// Index of a feature for a given king position and another piece on some square
template<Color Perspective>
IndexType Simplified_Threats::make_index(Piece attkr, Square from, Square to, Piece attkd, Square ksq) {
    bool enemy = ((attkr ^ attkd) & 8);
    from = (Square)(int(from) ^ OrientTBL[Perspective][ksq]);
    to = (Square)(int(to) ^ OrientTBL[Perspective][ksq]);
    if (Perspective == BLACK) {
        attkr = ~attkr;
        attkd = ~attkd;
    }
    if (from == to) {
        return IndexType(PieceSquareIndex[attkr]+from);
    }
    Bitboard attacks = (type_of(attkr) == PAWN) ? pawn_attacks_bb(color_of(attkr), from) : attacks_bb(type_of(attkr), from, 0ULL);
    return IndexType(768 + threatoffsets[attkr][65] + enemy*threatoffsets[attkr][64]
                     + threatoffsets[attkr][from] + popcount((square_bb(to)-1) & attacks));
}

// Get a list of indices for active features in ascending order
template<Color Perspective>
void Simplified_Threats::append_active_threats(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active) {
    Square ksq = lsb(colorBB[Perspective] & pieceBB[KING]);
    Color order[2][2] = {{WHITE, BLACK}, {BLACK, WHITE}};
    Bitboard occupied = colorBB[WHITE] | colorBB[BLACK];
    IndexList indices;

    for (int i = WHITE; i <= BLACK; i++) {
        for (int j = PAWN; j <= KING; j++) {
            Color c = order[Perspective][i];
            PieceType pt = PieceType(j);
            Piece attkr = make_piece(c, pt);
            Bitboard bb  = colorBB[c] & pieceBB[pt];
            indices.clear();
            while (bb)
            {
                Square from = pop_lsb(bb);
                Bitboard attacks = ((pt == PAWN) ? pawn_attacks_bb(c, from) : attacks_bb(pt, from, occupied)) & occupied;
                while (attacks) {
                    Square to = pop_lsb(attacks);
                    Piece attkd = board[to];
                    indices.push_back(make_index<Perspective>(attkr, from, to, attkd, ksq));
                }
            }
            std::sort(indices.begin(), indices.end());
            for (auto threat : indices) {
                active.push_back(threat);
            }
        }
    }
}

template<Color Perspective>
void Simplified_Threats::append_active_psq(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active) {
    Square ksq = lsb(colorBB[Perspective] & pieceBB[KING]);
    Color order[2][2] = {{WHITE, BLACK}, {BLACK, WHITE}};
    IndexList indices;

    for (int i = WHITE; i <= BLACK; i++) {
        for (int j = PAWN; j <= KING; j++) {
            Color c = order[Perspective][i];
            PieceType pt = PieceType(j);
            Piece pc = make_piece(c, pt);
            Bitboard bb  = colorBB[c] & pieceBB[pt];
            indices.clear();
            while (bb)
            {
                Square s = pop_lsb(bb);
                indices.push_back(make_index<Perspective>(pc, s, s, pc, ksq));
            }
            std::sort(indices.begin(), indices.end());
            for (auto threat : indices) {
                active.push_back(threat);
            }
        }
    }
}

template<Color Perspective>
void Simplified_Threats::append_active_features(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& psq, IndexList& threats) {
    Square ksq = lsb(colorBB[Perspective] & pieceBB[KING]);
    Color order[2][2] = {{WHITE, BLACK}, {BLACK, WHITE}};
    Bitboard occupied = colorBB[WHITE] | colorBB[BLACK];
    IndexList indices;
    for (int i = WHITE; i <= BLACK; i++) {
        for (int j = PAWN; j <= KING; j++) {
            Color c = order[Perspective][i];
            PieceType pt = PieceType(j);
            Piece attkr = make_piece(c, pt);
            Bitboard bb  = colorBB[c] & pieceBB[pt];
            indices.clear();
            while (bb)
            {
                Square from = pop_lsb(bb);
                psq.push_back(make_index<Perspective>(attkr, from, from, attkr, ksq));
                Bitboard attacks = ((pt == PAWN) ? pawn_attacks_bb(c, from) : attacks_bb(pt, from, occupied)) & occupied;
                while (attacks) {
                    Square to = pop_lsb(attacks);
                    Piece attkd = board[to];
                    indices.push_back(make_index<Perspective>(attkr, from, to, attkd, ksq));
                }
            }
            std::sort(indices.begin(), indices.end());
            for (auto threat : indices) {
                threats.push_back(threat);
            }
        }
    }
}



// Explicit template instantiations
template void Simplified_Threats::append_active_threats<WHITE>(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active);
template void Simplified_Threats::append_active_threats<BLACK>(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active);
template void Simplified_Threats::append_active_psq<WHITE>(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active);
template void Simplified_Threats::append_active_psq<BLACK>(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& active);
template void Simplified_Threats::append_active_features<WHITE>(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& psq, IndexList& threats);
template void Simplified_Threats::append_active_features<BLACK>(const Bitboard *colorBB, const Bitboard *pieceBB, const Piece *board, IndexList& psq, IndexList& threats);
template IndexType Simplified_Threats::make_index<WHITE>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);
template IndexType Simplified_Threats::make_index<BLACK>(Piece attkr, Square from, Square to, Piece attkd, Square ksq);

// Get a list of indices for recently changed features
template<Color Perspective>
void Simplified_Threats::append_changed_indices(Square            ksq,
                                         const DirtyPiece& dp,
                                         IndexList&        removed,
                                         IndexList&        added) {
    for (int i = 0; i < dp.dirty_num; ++i)
    {
        if (dp.from[i] != SQ_NONE)
            removed.push_back(make_index<Perspective>(dp.piece[i], dp.from[i], dp.from[i], dp.piece[i], ksq));
        if (dp.to[i] != SQ_NONE)
            added.push_back(make_index<Perspective>(dp.piece[i], dp.to[i], dp.to[i], dp.piece[i], ksq));
    }
    /*
    std::cout << "removed indices\n";
    for (auto index : removed) {
        std::cout << index << " ";
    }
    std::cout << "\nadded indices\n";
    for (auto index : added) {
        std::cout << index << " ";
    }
    */
}

// Explicit template instantiations
template void Simplified_Threats::append_changed_indices<WHITE>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);
template void Simplified_Threats::append_changed_indices<BLACK>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);

bool Simplified_Threats::requires_refresh(const StateInfo* st, Color perspective) {
    return (st->dirtyPiece.piece[0] == make_piece(perspective, KING)
    && OrientTBL[perspective][st->dirtyPiece.from[0]] != OrientTBL[perspective][st->dirtyPiece.to[0]]);
}
}  // namespace Stockfish::Eval::NNUE::Features


