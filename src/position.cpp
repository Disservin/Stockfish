/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

#include "position.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

#include "bitboard.h"
#include "misc.h"

using std::string;

namespace Stockfish {

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side, noPawns;
}

namespace {

constexpr std::string_view PieceToChar(" PNBRQK  pnbrqk");

constexpr Piece Pieces[] = {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                            B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING};
}  // namespace


// Returns an ASCII representation of the position
std::ostream& operator<<(std::ostream& os, const Position& pos) {

    os << "\n +---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

        os << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";
    }

    os << "   a   b   c   d   e   f   g   h\n"
       << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << " " << std::setfill(' ') << std::dec << "\nCheckers: ";


    // if (int(Tablebases::MaxCardinality) >= popcount(pos.pieces()) && !pos.can_castle(ANY_CASTLING))
    // {
    //     StateInfo st;
    //     ASSERT_ALIGNED(&st, Eval::NNUE::CacheLineSize);

    //     Position p;
    //     p.set(pos.fen(), pos.is_chess960(), &st);
    //     Tablebases::ProbeState s1, s2;
    //     Tablebases::WDLScore   wdl = Tablebases::probe_wdl(p, &s1);
    //     int                    dtz = Tablebases::probe_dtz(p, &s2);
    //     os << "\nTablebases WDL: " << std::setw(4) << wdl << " (" << s1 << ")"
    //        << "\nTablebases DTZ: " << std::setw(4) << dtz << " (" << s2 << ")";
    // }

    return os;
}


// Implements Marcel van Kervinck's cuckoo algorithm to detect repetition of positions
// for 3-fold repetition draws. The algorithm uses two hash tables with Zobrist hashes
// to allow fast detection of recurring positions. For details see:
// http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

// First and second hash functions for indexing the cuckoo tables
inline int H1(Key h) { return h & 0x1fff; }
inline int H2(Key h) { return (h >> 16) & 0x1fff; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the moves themselves
Key  cuckoo[8192];
Move cuckooMove[8192];


// Initializes at startup the various arrays used to compute hash keys
void Position::init() {

    PRNG rng(1070372);

    for (Piece pc : Pieces)
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            Zobrist::psq[pc][s] = rng.rand<Key>();

    for (File f = FILE_A; f <= FILE_H; ++f)
        Zobrist::enpassant[f] = rng.rand<Key>();

    for (int cr = NO_CASTLING; cr <= ANY_CASTLING; ++cr)
        Zobrist::castling[cr] = rng.rand<Key>();

    Zobrist::side    = rng.rand<Key>();
    Zobrist::noPawns = rng.rand<Key>();

    // Prepare the cuckoo tables
    std::memset(cuckoo, 0, sizeof(cuckoo));
    std::memset(cuckooMove, 0, sizeof(cuckooMove));
    [[maybe_unused]] int count = 0;
    for (Piece pc : Pieces)
        for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
            for (Square s2 = Square(s1 + 1); s2 <= SQ_H8; ++s2)
                if ((type_of(pc) != PAWN) && (attacks_bb(type_of(pc), s1, 0) & s2))
                {
                    Move move = Move(s1, s2);
                    Key  key  = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2] ^ Zobrist::side;
                    int  i    = H1(key);
                    while (true)
                    {
                        std::swap(cuckoo[i], key);
                        std::swap(cuckooMove[i], move);
                        if (move == Move::none())  // Arrived at empty slot?
                            break;
                        i = (i == H1(key)) ? H2(key) : H1(key);  // Push victim to alternative slot
                    }
                    count++;
                }
    assert(count == 3668);
}


// Initializes the position object with the given FEN string.
// This function is not very robust - make sure that input FENs are correct,
// this is assumed to be the responsibility of the GUI.
Position& Position::set(const string& fenStr, bool, StateInfo* si) {
    /*
   A FEN string defines a particular position using only the ASCII character set.

   A FEN string contains six fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 8 and ending with rank 1. Within each rank, the contents of each
      square are described from file A through file H. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("PNBRQK") whilst Black uses lowercase ("pnbrqk"). Blank squares are
      noted using digits 1 through 8 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Castling availability. If neither side can castle, this is "-". Otherwise,
      this has one or more letters: "K" (White can castle kingside), "Q" (White
      can castle queenside), "k" (Black can castle kingside), and/or "q" (Black
      can castle queenside).

   4) En passant target square (in algebraic notation). If there's no en passant
      target square, this is "-". If a pawn has just made a 2-square move, this
      is the position "behind" the pawn. Following X-FEN standard, this is recorded
      only if there is a pawn in position to make an en passant capture, and if
      there really is a pawn that might have advanced two squares.

   5) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   6) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/

    unsigned char      col, row, token;
    size_t             idx;
    Square             sq = SQ_A8;
    std::istringstream ss(fenStr);

    std::memset(this, 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    ss >> std::noskipws;

    // 1. Piece placement
    while ((ss >> token) && !isspace(token))
    {
        if (isdigit(token))
            sq += (token - '0') * EAST;  // Advance the given number of files

        else if (token == '/')
            sq += 2 * SOUTH;

        else if ((idx = PieceToChar.find(token)) != string::npos)
        {
            put_piece(Piece(idx), sq);
            ++sq;
        }
    }

    // 2. Active color
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;

    // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
    // Shredder-FEN that uses the letters of the columns on which the rooks began
    // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
    // if an inner rook is associated with the castling right, the castling tag is
    // replaced by the file letter of the involved rook, as for the Shredder-FEN.
    while ((ss >> token) && !isspace(token))
    {
        Square rsq;
        Color  c    = islower(token) ? BLACK : WHITE;
        Piece  rook = make_piece(c, ROOK);

        token = char(toupper(token));

        if (token == 'K')
            for (rsq = relative_square(c, SQ_H1); piece_on(rsq) != rook; --rsq)
            {}

        else if (token == 'Q')
            for (rsq = relative_square(c, SQ_A1); piece_on(rsq) != rook; ++rsq)
            {}

        else if (token >= 'A' && token <= 'H')
            rsq = make_square(File(token - 'A'), relative_rank(c, RANK_1));

        else
            continue;

        // set_castling_right(c, rsq);
    }

    // 4. En passant square.
    // Ignore if square is invalid or not on side to move relative rank 6.
    bool enpassant = false;

    if (((ss >> col) && (col >= 'a' && col <= 'h'))
        && ((ss >> row) && (row == (sideToMove == WHITE ? '6' : '3'))))
    {
        st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

        // En passant square will be considered only if
        // a) side to move have a pawn threatening epSquare
        // b) there is an enemy pawn in front of epSquare
        // c) there is no piece on epSquare or behind epSquare
        enpassant = pawn_attacks_bb(~sideToMove, st->epSquare) & pieces(sideToMove, PAWN)
                 && (pieces(~sideToMove, PAWN) & (st->epSquare + pawn_push(~sideToMove)))
                 && !(pieces() & (st->epSquare | (st->epSquare + pawn_push(sideToMove))));
    }

    if (!enpassant)
        st->epSquare = SQ_NONE;

    // 5-6. Halfmove clock and fullmove number
    // ss >> std::skipws >> std::string("0") >> gamePly;

    // Convert from fullmove starting from 1 to gamePly starting from 0,
    // handle also common incorrect FEN with fullmove = 0.
    // gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    // chess960 = isChess960;
    set_state();


    return *this;
}


// Helper function used to set castling
// rights given the corresponding color and the rook starting square.
void Position::set_castling_right(Color c, Square rfrom) {

    Square         kfrom = square<KING>(c);
    CastlingRights cr    = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    // Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
    // Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

    // castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & ~(kfrom | rfrom);
}


// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up
void Position::set_state() const {

    st->key = st->materialKey = 0;

    for (Bitboard b = pieces(); b;)
    {
        Square s  = pop_lsb(b);
        Piece  pc = piece_on(s);
        st->key ^= Zobrist::psq[pc][s];

        // if (type_of(pc) == PAWN)
        // st->pawnKey ^= Zobrist::psq[pc][s];

        // else if (type_of(pc) != KING)
        //     st->nonPawnMaterial[color_of(pc)] += PieceValue[pc];
    }

    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];

    for (Piece pc : Pieces)
        for (int cnt = 0; cnt < pieceCount[pc]; ++cnt)
            st->materialKey ^= Zobrist::psq[pc][cnt];
}


// Overload to initialize the position object with the given endgame code string
// like "KBPKN". It's mainly a helper to get the material key out of an endgame code.
Position& Position::set(const string& code, Color c, StateInfo* si) {

    assert(code[0] == 'K');

    string sides[] = {code.substr(code.find('K', 1)),                                // Weak
                      code.substr(0, std::min(code.find('v'), code.find('K', 1)))};  // Strong

    assert(sides[0].length() > 0 && sides[0].length() < 8);
    assert(sides[1].length() > 0 && sides[1].length() < 8);

    std::transform(sides[c].begin(), sides[c].end(), sides[c].begin(), tolower);

    string fenStr = "8/" + sides[0] + char(8 - sides[0].length() + '0') + "/8/8/8/8/" + sides[1]
                  + char(8 - sides[1].length() + '0') + "/8 w - - 0 10";

    return set(fenStr, false, si);
}


// Returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.
string Position::fen() const {

    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_8; r >= RANK_1; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
        {
            for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_H)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r > RANK_1)
            ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    // if (can_castle(WHITE_OO))
    //     ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OO))) : 'K');

    // if (can_castle(WHITE_OOO))
    //     ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OOO))) : 'Q');

    // if (can_castle(BLACK_OO))
    //     ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OO))) : 'k');

    // if (can_castle(BLACK_OOO))
    //     ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OOO))) : 'q');

    // if (!can_castle(ANY_CASTLING))
    //     ss << '-';

    return ss.str();
}


}  // namespace Stockfish
