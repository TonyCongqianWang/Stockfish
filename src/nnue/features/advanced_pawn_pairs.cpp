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

#include "advanced_pawn_pairs.h"

#include <algorithm>
#include "../../position.h"
#include "../../bitboard.h"

namespace Stockfish::Eval::NNUE::Features {

AdvancedPawnPairs::Snapshot AdvancedPawnPairs::snapshot(const Position& pos) {
    Snapshot s{};
    Bitboard partners = pos.pieces() & ~pos.pieces(PAWN, KING);
    if (popcount(partners) > 3)
        return s;
    s.pawns[WHITE] = pos.pieces(WHITE, PAWN) & (Rank6BB | Rank7BB);
    s.pawns[BLACK] = pos.pieces(BLACK, PAWN) & (Rank2BB | Rank3BB);
    if (!(s.pawns[WHITE] | s.pawns[BLACK]))
        return s;
    while (partners)
    {
        Square sq            = pop_lsb(partners);
        s.pieces[s.count]    = pos.piece_on(sq);
        s.squares[s.count++] = sq;
    }
    return s;
}

IndexType AdvancedPawnPairs::make_index(
  Color perspective, Color pawnColor, Square pawn, Piece partner, Square square, Square ksq) {
    const int orientation = OrientTBL[ksq] ^ (56 * perspective);
    pawn                  = Square(int(pawn) ^ orientation);
    square                = Square(int(square) ^ orientation);
    pawnColor             = Color(pawnColor ^ perspective);
    const int pawnId = int(pawnColor) * 16 + (int(relative_rank(pawnColor, pawn)) - int(RANK_6)) * 8
                     + int(file_of(pawn));
    const int pieceId = (color_of(partner) ^ perspective) * 4 + type_of(partner) - KNIGHT;
    return (pawnId * 8 + pieceId) * 64 + square;
}

namespace {
void append_snapshot(Color                              perspective,
                     Square                             ksq,
                     const AdvancedPawnPairs::Snapshot& s,
                     AdvancedPawnPairs::IndexList&      active) {
    for (Color c : {WHITE, BLACK})
        for (Bitboard pawns = s.pawns[c]; pawns;)
        {
            Square pawn = pop_lsb(pawns);
            for (int i = 0; i < s.count; ++i)
                active.push_back(AdvancedPawnPairs::make_index(perspective, c, pawn, s.pieces[i],
                                                               s.squares[i], ksq));
        }
}
}

void AdvancedPawnPairs::append_active_indices(Color           perspective,
                                              const Position& pos,
                                              IndexList&      active) {
    append_snapshot(perspective, pos.square<KING>(perspective), snapshot(pos), active);
}

void AdvancedPawnPairs::append_changed_indices(Color                         perspective,
                                               Square                        ksq,
                                               const DirtyAdvancedPawnPairs& diff,
                                               IndexList&                    removed,
                                               IndexList&                    added) {
    if (!diff.before.count && !diff.after.count)
        return;
    if (diff.before.count == diff.after.count && diff.before.pawns[WHITE] == diff.after.pawns[WHITE]
        && diff.before.pawns[BLACK] == diff.after.pawns[BLACK])
    {
        bool same = true;
        for (int i = 0; i < diff.before.count; ++i)
            same &= diff.before.pieces[i] == diff.after.pieces[i]
                 && diff.before.squares[i] == diff.after.squares[i];
        if (same)
            return;
    }
    IndexList before, after;
    append_snapshot(perspective, ksq, diff.before, before);
    append_snapshot(perspective, ksq, diff.after, after);
    std::array<u16, MaxActiveDimensions> oldRows, newRows;
    std::copy(before.begin(), before.end(), oldRows.begin());
    std::copy(after.begin(), after.end(), newRows.begin());
    std::sort(oldRows.begin(), oldRows.begin() + before.size());
    std::sort(newRows.begin(), newRows.begin() + after.size());
    auto       a = oldRows.begin(), b = newRows.begin();
    const auto aEnd = a + before.size(), bEnd = b + after.size();
    while (a != aEnd || b != bEnd)
        if (b == bEnd || (a != aEnd && *a < *b))
            removed.push_back(*a++);
        else if (a == aEnd || *b < *a)
            added.push_back(*b++);
        else
        {
            ++a;
            ++b;
        }
}

}  // namespace Stockfish::Eval::NNUE::Features
