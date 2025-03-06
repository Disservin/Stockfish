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

// Code for calculating NNUE evaluation function

#include "nnue_misc.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string_view>
#include <tuple>

#include "../position.h"
#include "../types.h"
#include "../uci.h"
#include "network.h"
#include "nnue_accumulator.h"

namespace Stockfish::Eval::NNUE {


constexpr std::string_view PieceToChar(" PNBRQK  pnbrqk");


namespace {
// Converts a Value into (centi)pawns and writes it in a buffer.
// The buffer must have capacity for at least 5 chars.
void format_cp_compact(Value v, char* buffer, const Position& pos) {

    buffer[0] = (v < 0 ? '-' : v > 0 ? '+' : ' ');

    int cp = std::abs(UCIEngine::to_cp(v, pos));
    if (cp >= 10000)
    {
        buffer[1] = '0' + cp / 10000;
        cp %= 10000;
        buffer[2] = '0' + cp / 1000;
        cp %= 1000;
        buffer[3] = '0' + cp / 100;
        buffer[4] = ' ';
    }
    else if (cp >= 1000)
    {
        buffer[1] = '0' + cp / 1000;
        cp %= 1000;
        buffer[2] = '0' + cp / 100;
        cp %= 100;
        buffer[3] = '.';
        buffer[4] = '0' + cp / 10;
    }
    else
    {
        buffer[1] = '0' + cp / 100;
        cp %= 100;
        buffer[2] = '.';
        buffer[3] = '0' + cp / 10;
        cp %= 10;
        buffer[4] = '0' + cp / 1;
    }
}


// Converts a Value into pawns, always keeping two decimals
void format_cp_aligned_dot(Value v, std::stringstream& stream, const Position& pos) {

    const double pawns = std::abs(0.01 * UCIEngine::to_cp(v, pos));

    stream << (v < 0   ? '-'
               : v > 0 ? '+'
                       : ' ')
           << std::setiosflags(std::ios::fixed) << std::setw(6) << std::setprecision(2) << pawns;
}
}


// Returns a string with the value of each piece on a board,
// and a table for (PSQT, Layers) values bucket by bucket.
std::string
trace(Position& pos, const Eval::NNUE::Networks& networks, Eval::NNUE::AccumulatorCaches& caches) {

    std::stringstream ss;

    char board[3 * 8 + 1][8 * 8 + 2];
    std::memset(board, ' ', sizeof(board));
    for (int row = 0; row < 3 * 8 + 1; ++row)
        board[row][8 * 8 + 1] = '\0';

    // A lambda to output one box of the board
    auto writeSquare = [&board, &pos](File file, Rank rank, Piece pc, Value value) {
        const int x = int(file) * 8;
        const int y = (7 - int(rank)) * 3;
        for (int i = 1; i < 8; ++i)
            board[y][x + i] = board[y + 3][x + i] = '-';
        for (int i = 1; i < 3; ++i)
            board[y + i][x] = board[y + i][x + 8] = '|';
        board[y][x] = board[y][x + 8] = board[y + 3][x + 8] = board[y + 3][x] = '+';
        if (pc != NO_PIECE)
            board[y + 1][x + 4] = PieceToChar[pc];
        if (is_valid(value))
            format_cp_compact(value, &board[y + 2][x + 2], pos);
    };

    // We estimate the value of each piece by doing a differential evaluation from
    // the current base eval, simulating the removal of the piece from its square.
    Value base = networks.big.evaluate(pos);
    base                    = pos.side_to_move() == WHITE ? base : -base;

    for (File f = FILE_A; f <= FILE_H; ++f)
        for (Rank r = RANK_1; r <= RANK_8; ++r)
        {
            Square sq = make_square(f, r);
            Piece  pc = pos.piece_on(sq);
            Value  v  = VALUE_NONE;

            if (pc != NO_PIECE && type_of(pc) != KING)
            {
                auto st = pos.state();

                pos.remove_piece(sq);

                Value eval = networks.big.evaluate(pos);
                eval                       = pos.side_to_move() == WHITE ? eval : -eval;
                v                          = base - eval;

                pos.put_piece(pc, sq);
            }

            writeSquare(f, r, pc, v);
        }

    ss << " NNUE derived piece values:\n";
    for (int row = 0; row < 3 * 8 + 1; ++row)
        ss << board[row] << '\n';
    ss << '\n';
    ss << networks.big.get_ft_stats();
    return ss.str();
}

void write_difference(Features::Simplified_Threats::IndexList& a1, Features::Simplified_Threats::IndexList& b1, Features::Simplified_Threats::IndexList& a2, Features::Simplified_Threats::IndexList& b2) {
    unsigned long long a = 0;
    unsigned long long b = 0;
    while (a < a1.size() && b < b1.size()) {
        if (a1[a] < b1[b]) {
            a2.push_back(a1[a]);
            a++;
        }
        else if (b1[b] < a1[a]) {
            b2.push_back(b1[b]);
            b++;
        }
        else {
            a++;
            b++;
        }
    }
    while (a < a1.size()) {
        a2.push_back(a1[a]);
        a++;
    }
    while (b < b1.size()) {
        b2.push_back(b1[b]);
        b++;
    }
}

}  // namespace Stockfish::Eval::NNUE
