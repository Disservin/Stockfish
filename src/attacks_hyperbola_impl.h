#ifndef ATTACKS_HYPERBOLA_IMPL_H_INCLUDED
#define ATTACKS_HYPERBOLA_IMPL_H_INCLUDED

static void init_magics(Magic magics[][2]) {
    for (Square s = SQ_A1; s <= SQ_H8; ++s)
    {
        Magic& rook = magics[s][ROOK - BISHOP];
        rook.mask1  = line_mask(s, NORTH, SOUTH);
        rook.mask2  = line_mask(s, EAST, WEST);

        Magic& bishop = magics[s][BISHOP - BISHOP];
        bishop.mask1  = line_mask(s, NORTH_EAST, SOUTH_WEST);
        bishop.mask2  = line_mask(s, NORTH_WEST, SOUTH_EAST);

        rook.r = bishop.r = square_bb(s) * 2;
        rook.rr = bishop.rr = square_bb(Square(63 - int(s))) * 2;
    }
}

#endif  // ATTACKS_HYPERBOLA_IMPL_H_INCLUDED
