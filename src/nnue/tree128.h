// Frozen tree128 adapter. Routing and extraction are generated/shared with trainer.
#pragma once
#include "../position.h"
#include "tree128_generated.h"
#include "tree128_context.h"
namespace Stockfish::Eval::NNUE {
struct Tree128Board {
    const Position& pos;
    Color color(int side) const { return side ? ~pos.side_to_move() : pos.side_to_move(); }
    bool white_to_move() const { return pos.side_to_move() == WHITE; }
    int occupied(int side) const { return popcount(pos.pieces(color(side))); }
    int count(int side, int type) const { return popcount(pos.pieces(color(side), PieceType(type+1))); }
    std::uint64_t pieces(int side, int type) const { return pos.pieces(color(side), PieceType(type+1)); }
    int king(int side) const { return int(pos.square<KING>(color(side))); }
};
inline int tree128(const Position& pos) {
    return frozen_tree128::select(frozen_tree128::Context<Tree128Board>(Tree128Board{pos}));
}
}
