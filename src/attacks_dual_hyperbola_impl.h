#ifndef ATTACKS_DUAL_HYPERBOLA_IMPL_H_INCLUDED
#define ATTACKS_DUAL_HYPERBOLA_IMPL_H_INCLUDED

// Sliding attacks within a rank, indexed by the slider's file and the
// 8-bit rank occupancy, yielding the 8-bit attack set on that rank.
constexpr auto RankAttacks = []() {
    std::array<std::array<uint8_t, 256>, FILE_NB> table{};
    for (int file = 0; file < 8; ++file)
        for (int occ = 0; occ < 256; ++occ)
        {
            uint8_t attacks = 0;
            for (int f = file + 1; f <= 7; ++f)
            {
                attacks |= uint8_t(1 << f);
                if (occ & (1 << f))
                    break;
            }
            for (int f = file - 1; f >= 0; --f)
            {
                attacks |= uint8_t(1 << f);
                if (occ & (1 << f))
                    break;
            }
            table[file][occ] = attacks;
        }
    return table;
}();

static void init_dual_magics(DualMagic magics[]) {
    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        DualMagic& m        = magics[s];
        m.maskFile          = line_mask(s, NORTH, SOUTH);
        m.maskDiag          = line_mask(s, NORTH_EAST, SOUTH_WEST);
        m.maskNone          = 0;
        m.maskAntidiag      = line_mask(s, NORTH_WEST, SOUTH_EAST);
        m.r                 = square_bb(s) * 2;
        m.rr                = square_bb(Square(63 - int(s))) * 2;
        m.rankAttacksLookup = RankAttacks[int(file_of(s))].data();
        m.shift             = 8 * int(rank_of(s));
    }
}

#endif  // ATTACKS_DUAL_HYPERBOLA_IMPL_H_INCLUDED
