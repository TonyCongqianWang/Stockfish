/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  SPDX-License-Identifier: GPL-3.0-or-later
*/
#include "pawn_structure.h"

#include <algorithm>
#include <array>
#include "../../bitboard.h"
#include "../../position.h"

namespace Stockfish::Eval::NNUE::Features {
namespace {
int reflect(int rank) { return rank ? 7 - rank : 0; }
int state(Color p, int f, Bitboard white, Bitboard black) {
    white &= file_bb(File(f));
    black &= file_bb(File(f));
    int w = white ? rank_of(msb(white)) : 0;
    int b = black ? rank_of(lsb(black)) : 0;
    return p == WHITE ? 7 * w + b : 7 * reflect(b) + reflect(w);
}
using FilePairs = ValueList<u16, 6>;
FilePairs pairs(Color p, int file, Bitboard white, Bitboard black, Square ksq) {
    FilePairs out;
    Bitboard  occupied = (white | black) & file_bb(File(file));
    int       previous = -1;
    while (occupied)
    {
        Square sq = pop_lsb(occupied);
        int    id = ((black & square_bb(sq)) ? 6 : 0) + rank_of(sq) - 1;
        if (previous >= 0)
            out.push_back(PawnStructure::file_pair_index(p, file, previous, id, ksq));
        previous = id;
    }
    return out;
}
int rank_pair(int a, int b) {
    if (a > b)
        std::swap(a, b);
    return a * (11 - a) / 2 + b - a - 1;
}
}  // namespace

IndexType
PawnStructure::frontier_index(Color p, int file, Bitboard white, Bitboard black, Square ksq) {
    if (ksq != SQ_NONE && !(int(ksq) & 4))
        return IndexBase + 2401 * (6 - file) + 49 * state(p, file + 1, white, black)
             + state(p, file, white, black);
    return IndexBase + 2401 * file + 49 * state(p, file, white, black)
         + state(p, file + 1, white, black);
}

IndexType PawnStructure::file_pair_index(Color p, int file, int a, int b, Square ksq) {
    if (ksq != SQ_NONE && !(int(ksq) & 4))
        file = 7 - file;
    if (p == BLACK)
    {
        a = (1 - a / 6) * 6 + 5 - a % 6;
        b = (1 - b / 6) * 6 + 5 - b % 6;
    }
    int index;
    if (a / 6 == b / 6)
        index = 15 * (a / 6) + rank_pair(a % 6, b % 6);
    else
    {
        if (a >= 6)
            std::swap(a, b);
        b -= 6;
        index = 30 + 5 * a + b - (b > a);
    }
    return IndexBase + FrontierDimensions + 60 * file + index;
}

void PawnStructure::append_active_indices(Color p, const Position& pos, IndexList& active) {
    Bitboard white = pos.pieces(WHITE, PAWN), black = pos.pieces(BLACK, PAWN);
    for (int f = 0; f < 7; ++f)
        active.push_back(frontier_index(p, f, white, black, pos.square<KING>(p)));
    for (int f = 0; f < 8; ++f)
        for (auto i : pairs(p, f, white, black, pos.square<KING>(p)))
            active.push_back(i);
}

void PawnStructure::append_changed_indices(Color           p,
                                           Square          ksq,
                                           const DiffType& d,
                                           IndexList&      removed,
                                           IndexList&      added,
                                           const ThreatWeightType*,
                                           IndexType) {
    Bitboard changed = (d.before[WHITE] ^ d.after[WHITE]) | (d.before[BLACK] ^ d.after[BLACK]);
    if (!changed)
        return;
    unsigned files = 0, boundaries = 0;
    while (changed)
        files |= 1u << file_of(pop_lsb(changed));
    boundaries = (files | (files >> 1)) & 127u;
    for (int f = 0; f < 7; ++f)
        if (boundaries & (1u << f))
        {
            auto before = frontier_index(p, f, d.before[WHITE], d.before[BLACK], ksq);
            auto after  = frontier_index(p, f, d.after[WHITE], d.after[BLACK], ksq);
            if (before != after)
            {
                removed.push_back(before);
                added.push_back(after);
            }
        }
    for (int f = 0; f < 8; ++f)
        if (files & (1u << f))
        {
            auto before = pairs(p, f, d.before[WHITE], d.before[BLACK], ksq);
            auto after  = pairs(p, f, d.after[WHITE], d.after[BLACK], ksq);
            for (auto i : before)
                if (std::find(after.begin(), after.end(), i) == after.end())
                    removed.push_back(i);
            for (auto i : after)
                if (std::find(before.begin(), before.end(), i) == before.end())
                    added.push_back(i);
        }
}

void PawnStructure::append_changed_indices_both(Square          wk,
                                                Square          bk,
                                                const DiffType& d,
                                                IndexList&      wr,
                                                IndexList&      wa,
                                                IndexList&      br,
                                                IndexList&      ba,
                                                const ThreatWeightType*,
                                                IndexType) {
    append_changed_indices(WHITE, wk, d, wr, wa);
    append_changed_indices(BLACK, bk, d, br, ba);
}
}  // namespace Stockfish::Eval::NNUE::Features
