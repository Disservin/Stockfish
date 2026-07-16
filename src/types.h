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

#ifndef TYPES_H_INCLUDED
    #define TYPES_H_INCLUDED

// When compiling with provided Makefile (e.g. for Linux and OSX), configuration
// is done automatically. To get started type 'make help'.
//
// When Makefile is not used (e.g. with Microsoft Visual Studio) some switches
// need to be set manually:
//
// -DNDEBUG      | Disable debugging mode. Always use this for release.
//
// -DNO_PREFETCH | Disable use of prefetch asm-instruction. You may need this to
//               | run on some very old machines.
//
// -DUSE_POPCNT  | Add runtime support for use of popcnt asm-instruction. Works
//               | only in 64-bit mode and requires hardware with popcnt support.
//
// -DUSE_PEXT    | Add runtime support for use of pext asm-instruction. Works
//               | only in 64-bit mode and requires hardware with pext support.

    #include <cassert>
    #include <cstddef>
    #include <cstdint>
    #include "misc.h"

    #if defined(_MSC_VER)
        // Disable some silly and noisy warnings from MSVC compiler
        #pragma warning(disable: 4127)  // Conditional expression is constant
        #pragma warning(disable: 4146)  // Unary minus operator applied to unsigned type
        #pragma warning(disable: 4800)  // Forcing value to bool 'true' or 'false'
    #endif

// Predefined macros hell:
//
// __GNUC__                Compiler is GCC, Clang or ICX
// __clang__               Compiler is Clang or ICX
// __INTEL_LLVM_COMPILER   Compiler is ICX
// _MSC_VER                Compiler is MSVC
// _WIN32                  Building on Windows (any)
// _WIN64                  Building on Windows 64 bit

    // Enforce minimum GCC version
    #if defined(__GNUC__) && !defined(__clang__) \
      && (__GNUC__ < 9 || (__GNUC__ == 9 && __GNUC_MINOR__ < 3))
        #error "Stockfish requires GCC 9.3 or later for correct compilation"
    #endif

    // Enforce minimum Clang version
    #if defined(__clang__) && (__clang_major__ < 11)
        #error "Stockfish requires Clang 11.0 or later for correct compilation"
    #endif

    #define ASSERT_ALIGNED(ptr, alignment) assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0)

    #if defined(_WIN64) && defined(_MSC_VER)  // No Makefile used
        #include <intrin.h>                   // Microsoft header for _BitScanForward64()
        #define IS_64BIT
    #endif

    #if defined(_MSC_VER)
        #include <nmmintrin.h>  // Microsoft header for _mm_popcnt_u64()
    #endif

    #if !defined(NO_PREFETCH) && defined(_MSC_VER)
        #include <xmmintrin.h>  // Microsoft header for _mm_prefetch()
    #endif

    #if defined(USE_PEXT)
        #include <immintrin.h>  // Header for _pext_u64() intrinsic
        #define pext(b, m) _pext_u64(b, m)
        #define pdep(b, m) _pdep_u64(b, m)
    #endif

namespace Stockfish {

    #ifdef USE_POPCNT
constexpr bool HasPopCnt = true;
    #else
constexpr bool HasPopCnt = false;
    #endif

    #ifdef USE_PEXT
constexpr bool HasPext = true;
    #else
constexpr bool HasPext = false;
    #endif

    #ifdef IS_64BIT
constexpr bool Is64Bit = true;
    #else
constexpr bool Is64Bit = false;
    #endif

using Key      = u64;
using Bitboard = u64;

constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY   = 246;

enum Color : u8 {
    WHITE,
    BLACK,
    COLOR_NB = 2
};

enum CastlingRights : u8 {
    NO_CASTLING,
    WHITE_OO,
    WHITE_OOO = WHITE_OO << 1,
    BLACK_OO  = WHITE_OO << 2,
    BLACK_OOO = WHITE_OO << 3,

    KING_SIDE      = WHITE_OO | BLACK_OO,
    QUEEN_SIDE     = WHITE_OOO | BLACK_OOO,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,

    CASTLING_RIGHT_NB = 16
};

enum Bound : u8 {
    BOUND_NONE,
    BOUND_UPPER,
    BOUND_LOWER,
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

// Value is used as an alias for int, this is done to differentiate between a search
// value and any other integer value. The values used in search are always supposed
// to be in the range (-VALUE_NONE, VALUE_NONE] and should not exceed this range.
using Value = int;

constexpr Value VALUE_ZERO     = 0;
constexpr Value VALUE_DRAW     = 0;
constexpr Value VALUE_NONE     = 32002;
constexpr Value VALUE_INFINITE = 32001;

constexpr Value VALUE_MATE             = 32000;
constexpr Value VALUE_MATE_IN_MAX_PLY  = VALUE_MATE - MAX_PLY;
constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

constexpr Value VALUE_TB                 = VALUE_MATE_IN_MAX_PLY - 1;
constexpr Value VALUE_TB_WIN_IN_MAX_PLY  = VALUE_TB - MAX_PLY;
constexpr Value VALUE_TB_LOSS_IN_MAX_PLY = -VALUE_TB_WIN_IN_MAX_PLY;


constexpr bool is_valid(Value value) { return value != VALUE_NONE; }

constexpr bool is_win(Value value) {
    assert(is_valid(value));
    return value >= VALUE_TB_WIN_IN_MAX_PLY;
}

constexpr bool is_loss(Value value) {
    assert(is_valid(value));
    return value <= VALUE_TB_LOSS_IN_MAX_PLY;
}

constexpr bool is_decisive(Value value) { return is_win(value) || is_loss(value); }

constexpr bool is_mate(Value value) {
    assert(is_valid(value));
    return value >= VALUE_MATE_IN_MAX_PLY;
}

constexpr bool is_mated(Value value) {
    assert(is_valid(value));
    return value <= VALUE_MATED_IN_MAX_PLY;
}

constexpr bool is_mate_or_mated(Value value) { return is_mate(value) || is_mated(value); }

constexpr Value mate_in(int ply) { return VALUE_MATE - ply; }

constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }

// In the code, we make the assumption that these values
// are such that non_pawn_material() can be used to uniquely
// identify the material on the board.
constexpr Value PawnValue   = 208;
constexpr Value KnightValue = 781;
constexpr Value BishopValue = 825;
constexpr Value RookValue   = 1276;
constexpr Value QueenValue  = 2538;


// clang-format off
enum PieceType : u8 {
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    ALL_PIECES = 0,
    PIECE_TYPE_NB = 8
};

enum Piece : u8 {
    NO_PIECE,
    W_PAWN = PAWN,     W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = PAWN + 8, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};
// clang-format on

constexpr Value PieceValue[PIECE_NB] = {
  VALUE_ZERO, PawnValue, KnightValue, BishopValue, RookValue, QueenValue, VALUE_ZERO, VALUE_ZERO,
  VALUE_ZERO, PawnValue, KnightValue, BishopValue, RookValue, QueenValue, VALUE_ZERO, VALUE_ZERO};

using Depth = int;

// The following DEPTH_ constants are used for transposition table entries
// and quiescence search move generation stages. In regular search, the
// depth stored in the transposition table is literal: the search depth
// (effort) used to make the corresponding transposition table value. In
// quiescence search, however, the transposition table entries only store
// the current quiescence move generation stage (which should thus compare
// lower than any regular search depth).
constexpr Depth DEPTH_QS = 0;
// For transposition table entries where no searching at all was done
// (whether regular or qsearch) we use DEPTH_UNSEARCHED, which should thus
// compare lower than any quiescence or regular depth. DEPTH_NONE is used
// for the transposition table entry occupancy check (see tt.cpp), and
// should thus be lower than DEPTH_UNSEARCHED.
constexpr Depth DEPTH_UNSEARCHED = -2;
constexpr Depth DEPTH_NONE       = -3;

enum File : u8;
enum Rank : u8;

    #if defined NDEBUG
        #define X_ASSERT(CHECK) void(0)
    #else
        #define X_ASSERT(CHECK) ((CHECK) ? void(0) : [] { assert(!#CHECK); }())
    #endif

class Square {
   public:
    Square() = default;
    constexpr Square(int s) :
        value(u8(s)) {
        X_ASSERT(within_range());
    }

    constexpr operator int() const {
        X_ASSERT(within_range());
        return value;
    }

    constexpr bool   is_ok() const { return value < 64; }
    constexpr bool   within_range() const { return value < 65; }
    constexpr Square flip_rank() const;
    constexpr Square flip_file() const;
    constexpr File   file() const;
    constexpr Rank   rank() const;
    constexpr Square relative(Color c) const;

   private:
    u8 value;
};

static_assert(sizeof(Square) == sizeof(u8));

    #define SQ(Name, Value) inline constexpr Square Name(Value)

// clang-format off
SQ(SQ_A1,  0); SQ(SQ_B1,  1); SQ(SQ_C1,  2); SQ(SQ_D1,  3); SQ(SQ_E1,  4); SQ(SQ_F1,  5); SQ(SQ_G1,  6); SQ(SQ_H1,  7);
SQ(SQ_A2,  8); SQ(SQ_B2,  9); SQ(SQ_C2, 10); SQ(SQ_D2, 11); SQ(SQ_E2, 12); SQ(SQ_F2, 13); SQ(SQ_G2, 14); SQ(SQ_H2, 15);
SQ(SQ_A3, 16); SQ(SQ_B3, 17); SQ(SQ_C3, 18); SQ(SQ_D3, 19); SQ(SQ_E3, 20); SQ(SQ_F3, 21); SQ(SQ_G3, 22); SQ(SQ_H3, 23);
SQ(SQ_A4, 24); SQ(SQ_B4, 25); SQ(SQ_C4, 26); SQ(SQ_D4, 27); SQ(SQ_E4, 28); SQ(SQ_F4, 29); SQ(SQ_G4, 30); SQ(SQ_H4, 31);
SQ(SQ_A5, 32); SQ(SQ_B5, 33); SQ(SQ_C5, 34); SQ(SQ_D5, 35); SQ(SQ_E5, 36); SQ(SQ_F5, 37); SQ(SQ_G5, 38); SQ(SQ_H5, 39);
SQ(SQ_A6, 40); SQ(SQ_B6, 41); SQ(SQ_C6, 42); SQ(SQ_D6, 43); SQ(SQ_E6, 44); SQ(SQ_F6, 45); SQ(SQ_G6, 46); SQ(SQ_H6, 47);
SQ(SQ_A7, 48); SQ(SQ_B7, 49); SQ(SQ_C7, 50); SQ(SQ_D7, 51); SQ(SQ_E7, 52); SQ(SQ_F7, 53); SQ(SQ_G7, 54); SQ(SQ_H7, 55);
SQ(SQ_A8, 56); SQ(SQ_B8, 57); SQ(SQ_C8, 58); SQ(SQ_D8, 59); SQ(SQ_E8, 60); SQ(SQ_F8, 61); SQ(SQ_G8, 62); SQ(SQ_H8, 63);
// clang-format on

SQ(SQ_NONE, 64);
SQ(SQUARE_ZERO, 0);

inline constexpr int SQUARE_NB = 64;

    #undef SQ

enum Direction : i8 {
    NORTH = 8,
    EAST  = 1,
    SOUTH = -NORTH,
    WEST  = -EAST,

    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};

enum File : u8 {
    FILE_A,
    FILE_B,
    FILE_C,
    FILE_D,
    FILE_E,
    FILE_F,
    FILE_G,
    FILE_H,
    FILE_NB
};

enum Rank : u8 {
    RANK_1,
    RANK_2,
    RANK_3,
    RANK_4,
    RANK_5,
    RANK_6,
    RANK_7,
    RANK_8,
    RANK_NB
};

constexpr Square Square::flip_rank() const { return Square(*this ^ SQ_A8); }

constexpr Square Square::flip_file() const { return Square(*this ^ SQ_H1); }

constexpr File Square::file() const { return File(*this & 7); }

constexpr Rank Square::rank() const { return Rank(*this >> 3); }

constexpr Square Square::relative(Color c) const { return Square(*this ^ (c * 56)); }

// Keep track of what a move changes on the board (used by NNUE)
struct DirtyPiece {
    Piece  pc;        // this is never allowed to be NO_PIECE
    Square from, to;  // to should be SQ_NONE for promotions

    // if {add,remove}_sq is SQ_NONE, {add,remove}_pc is allowed to be
    // uninitialized
    // castling uses add_sq and remove_sq to remove and add the rook
    Square remove_sq, add_sq;
    Piece  remove_pc, add_pc;
};

// Keep track of what threats change on the board (used by NNUE)
struct DirtyThreat {
    static constexpr int PcSqOffset         = 0;
    static constexpr int ThreatenedSqOffset = 8;
    static constexpr int ThreatenedPcOffset = 16;
    static constexpr int PcOffset           = 20;

    DirtyThreat() { /* don't initialize data */ }
    DirtyThreat(u32 raw) :
        data(raw) {}
    DirtyThreat(Piece pc, Piece threatened_pc, Square pc_sq, Square threatened_sq, bool add) {
        data = (u32(add) << 31) | (pc << PcOffset) | (threatened_pc << ThreatenedPcOffset)
             | (threatened_sq << ThreatenedSqOffset) | (pc_sq << PcSqOffset);
    }

    Piece  pc() const { return static_cast<Piece>(data >> PcOffset & 0xf); }
    Piece  threatened_pc() const { return static_cast<Piece>(data >> ThreatenedPcOffset & 0xf); }
    Square threatened_sq() const { return static_cast<Square>(data >> ThreatenedSqOffset & 0xff); }
    Square pc_sq() const { return static_cast<Square>(data >> PcSqOffset & 0xff); }
    bool   add() const { return data >> 31; }
    u32    raw() const { return data; }

   private:
    u32 data;
};

// A piece can be involved in at most 8 outgoing attacks and 16 incoming attacks.
// Moving a piece also can reveal at most 8 discovered attacks.
// This implies that a non-castling move can change at most (8 + 16) * 3 + 8 = 80 features.
// By similar logic, a castling move can change at most (5 + 1 + 3 + 9) * 2 = 36 features.
// Thus, 80 should work as an upper bound. Finally, 16 entries are added to accommodate
// unmasked vector stores near the end of the list.

using DirtyThreatList = ValueList<DirtyThreat, 96>;

struct DirtyThreats {
    DirtyThreatList list;
};

    #define ENABLE_INCR_OPERATORS_ON(T) \
        constexpr T& operator++(T& d) { return d = T(int(d) + 1); } \
        constexpr T& operator--(T& d) { return d = T(int(d) - 1); }

ENABLE_INCR_OPERATORS_ON(PieceType)
ENABLE_INCR_OPERATORS_ON(Square)
ENABLE_INCR_OPERATORS_ON(File)
ENABLE_INCR_OPERATORS_ON(Rank)

    #undef ENABLE_INCR_OPERATORS_ON

constexpr Direction operator+(Direction d1, Direction d2) { return Direction(int(d1) + int(d2)); }
constexpr Direction operator*(int i, Direction d) { return Direction(i * int(d)); }

// Additional operators to add a Direction to a Square
constexpr Square  operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square  operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Square& operator+=(Square& s, Direction d) { return s = s + d; }
constexpr Square& operator-=(Square& s, Direction d) { return s = s - d; }

// Toggle color
constexpr Color operator~(Color c) { return Color(c ^ BLACK); }

// Swap color of piece B_KNIGHT <-> W_KNIGHT
constexpr Piece operator~(Piece pc) { return Piece(pc ^ 8); }

constexpr CastlingRights operator&(Color c, CastlingRights cr) {
    return CastlingRights((c == WHITE ? WHITE_CASTLING : BLACK_CASTLING) & cr);
}

constexpr Square make_square(File f, Rank r) { return Square((r << 3) + f); }

constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) + pt); }

constexpr PieceType type_of(Piece pc) { return PieceType(pc & 7); }

constexpr Color color_of(Piece pc) {
    assert(pc != NO_PIECE);
    return Color(pc >> 3);
}

constexpr Rank relative_rank(Color c, Rank r) { return Rank(r ^ (c * 7)); }

constexpr Direction pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }


// Based on a congruential pseudo-random number generator
constexpr Key make_key(u64 seed) { return seed * 6364136223846793005ULL + 1442695040888963407ULL; }


enum MoveType : u16 {
    NORMAL,
    PROMOTION  = 1 << 14,
    EN_PASSANT = 2 << 14,
    CASTLING   = 3 << 14
};

// A move needs 16 bits to be stored
//
// bit  0- 5: destination square (from 0 to 63)
// bit  6-11: origin square (from 0 to 63)
// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
// NOTE: en passant bit is set only when a pawn can be captured
//
// Special cases are Move::none() and Move::null(). We can sneak these in because
// in any normal move the destination square and origin square are always different,
// but Move::none() and Move::null() have the same origin and destination square.

class Move {
   public:
    Move() = default;
    constexpr explicit Move(u16 d) :
        data(d) {}

    constexpr Move(Square from, Square to) :
        data((from << 6) + to) {}

    template<MoveType T>
    static constexpr Move make(Square from, Square to, PieceType pt = KNIGHT) {
        return Move(T + ((pt - KNIGHT) << 12) + (from << 6) + to);
    }

    constexpr Square from_sq() const {
        assert(is_ok());
        return Square((data >> 6) & 0x3F);
    }

    constexpr Square to_sq() const {
        assert(is_ok());
        return Square(data & 0x3F);
    }

    constexpr MoveType type_of() const { return MoveType(data & (3 << 14)); }

    constexpr PieceType promotion_type() const { return PieceType(((data >> 12) & 3) + KNIGHT); }

    constexpr bool is_ok() const { return none().data != data && null().data != data; }

    static constexpr Move null() { return Move(65); }
    static constexpr Move none() { return Move(0); }

    constexpr bool operator==(const Move& m) const { return data == m.data; }
    constexpr bool operator!=(const Move& m) const { return data != m.data; }

    constexpr explicit operator bool() const { return data != 0; }

    constexpr u16 raw() const { return data; }

    struct MoveHash {
        usize operator()(const Move& m) const { return make_key(m.data); }
    };

    static constexpr int FromSqShift = 6;
    static constexpr int ToSqShift   = 0;

   protected:
    u16 data;
};

}  // namespace Stockfish

#endif  // #ifndef TYPES_H_INCLUDED

#include "tune.h"  // Global visibility to tuning setup
