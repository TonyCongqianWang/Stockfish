/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  SPDX-License-Identifier: GPL-3.0-or-later
*/

#include "global_threats.h"

#include <algorithm>
#include <array>

#include "../../attacks.h"
#include "../../bitboard.h"
#include "../../position.h"

namespace Stockfish::Eval::NNUE::Features {
namespace {

// Preserve upstream's relation selection, including the reciprocal geometry
// exclusions P->B/Q and B/R->Q. Pawn-pawn structure is a separate input family.
constexpr int TargetMap[5][5] = {
  {-1, 0, -1, 1, -1}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, -1}, {0, 1, 2, 3, -1}, {0, 1, 2, 3, 4}};
constexpr int Slots[6] = {0, 84, 336, 560, 896, 1456};
constexpr int Bases[6] = {0, 0, 168, 1848, 4088, 7672};

constexpr auto make_geometry() {
    std::array<std::array<std::array<u16, 64>, 64>, 6> result{};
    for (int pt = PAWN; pt <= QUEEN; ++pt)
    {
        int offset = 0;
        for (int from = 0; from < 64; ++from)
        {
            for (int to = 0; to < 64; ++to)
            {
                Bitboard attacks     = pt == PAWN
                                       ? Attacks::PseudoAttacks[to < from ? BLACK : WHITE][from]
                                       : Attacks::PseudoAttacks[pt][from];
                result[pt][from][to] = offset + constexpr_popcount(((1ULL << to) - 1) & attacks);
            }
            if (pt != PAWN || (from >= SQ_A2 && from <= SQ_H7))
                offset +=
                  constexpr_popcount(Attacks::PseudoAttacks[pt == PAWN ? int(WHITE) : pt][from]);
        }
    }
    return result;
}
constexpr auto Geometry = make_geometry();

bool selected(Piece a, Square from, Square to, Piece b) {
    return type_of(a) < KING && type_of(b) < KING && (type_of(a) != type_of(b) || from > to);
}

IndexType relation_index(Piece a, Square from, Square to, Piece b) {
    const int pt = type_of(a), target = type_of(b);
    if (pt < PAWN || pt >= KING || target < PAWN || target >= KING)
        return GlobalSupports::Dimensions;
    const int mapped = TargetMap[pt - 1][target - 1];
    if (mapped < 0)
        return GlobalSupports::Dimensions;
    if (pt == target && from < to)
        std::swap(from, to);
    return Bases[pt] + mapped * Slots[pt] + Geometry[pt][from][to];
}

template<typename F>
void visit_active(const Position& pos, F emit) {
    const Bitboard occupied = pos.pieces();
    for (Color c : {WHITE, BLACK})
        for (PieceType pt = PAWN; pt < KING; ++pt)
        {
            Bitboard pieces = pos.pieces(c, pt);
            while (pieces)
            {
                Square   from    = pop_lsb(pieces);
                Bitboard targets = (pt == PAWN ? Attacks::PseudoAttacks[c][from]
                                               : Attacks::attacks_bb(pt, from, occupied))
                                 & occupied;
                while (targets)
                {
                    Square to = pop_lsb(targets);
                    Piece  a = make_piece(c, pt), b = pos.piece_on(to);
                    if (selected(a, from, to, b))
                        emit(a, from, to, b);
                }
            }
        }
}
}  // namespace

IndexType
GlobalSupports::make_index(Color p, Piece a, Square from, Square to, Piece b, Square ksq) {
    if (color_of(a) != color_of(b))
        return Dimensions;
    const int orientation = 56 * p + (ksq != SQ_NONE && !(int(ksq) & 4) ? 7 : 0);
    return relation_index(a, Square(int(from) ^ orientation), Square(int(to) ^ orientation), b);
}

IndexType
GlobalAttacks::make_index(Color perspective, Piece a, Square from, Square to, Piece b, Square ksq) {
    if (color_of(a) == color_of(b))
        return Dimensions;
    const int orientation = 56 * perspective + (ksq != SQ_NONE && !(int(ksq) & 4) ? 7 : 0);
    from                  = Square(int(from) ^ orientation);
    to                    = Square(int(to) ^ orientation);
    a                     = Piece(a ^ (8 * perspective));
    b                     = Piece(b ^ (8 * perspective));
    if (type_of(a) == type_of(b) && from < to)
    {
        std::swap(a, b);
        std::swap(from, to);
    }
    IndexType index = relation_index(a, from, to, b);
    return index < GlobalSupports::Dimensions ? color_of(a) * GlobalSupports::Dimensions + index
                                              : Dimensions;
}

void GlobalSupports::append_active_indices(Color           p,
                                           bool            enemy,
                                           const Position& pos,
                                           IndexList&      active) {
    visit_active(pos, [&](Piece a, Square from, Square to, Piece b) {
        if ((color_of(a) != p) == enemy)
            active.push_back_if_lt(make_index(p, a, from, to, b, pos.square<KING>(p)), Dimensions);
    });
}

void GlobalSupports::append_changed_indices(
  Color p, bool enemy, Square ksq, const DirtyThreats& diff, IndexList& removed, IndexList& added) {
    for (const auto& d : diff.list)
        if ((color_of(d.pc()) != p) == enemy
            && selected(d.pc(), d.pc_sq(), d.threatened_sq(), d.threatened_pc()))
            (d.add() ? added : removed)
              .push_back_if_lt(
                make_index(p, d.pc(), d.pc_sq(), d.threatened_sq(), d.threatened_pc(), ksq),
                Dimensions);
}

void GlobalAttacks::append_active_indices(Color p, const Position& pos, IndexList& active) {
    visit_active(pos, [&](Piece a, Square from, Square to, Piece b) {
        active.push_back_if_lt(make_index(p, a, from, to, b, pos.square<KING>(p)), Dimensions);
    });
}

void GlobalAttacks::append_changed_indices(Color           p,
                                           Square          ksq,
                                           const DiffType& diff,
                                           IndexList&      removed,
                                           IndexList&      added,
                                           const ThreatWeightType*,
                                           IndexType) {
    for (const auto& d : diff.list)
        if (selected(d.pc(), d.pc_sq(), d.threatened_sq(), d.threatened_pc()))
            (d.add() ? added : removed)
              .push_back_if_lt(
                make_index(p, d.pc(), d.pc_sq(), d.threatened_sq(), d.threatened_pc(), ksq),
                Dimensions);
}

void GlobalAttacks::append_changed_indices_both(Square          wk,
                                                Square          bk,
                                                const DiffType& diff,
                                                IndexList&      wr,
                                                IndexList&      wa,
                                                IndexList&      br,
                                                IndexList&      ba,
                                                const ThreatWeightType*,
                                                IndexType) {
    append_changed_indices(WHITE, wk, diff, wr, wa);
    append_changed_indices(BLACK, bk, diff, br, ba);
}
}  // namespace Stockfish::Eval::NNUE::Features
