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

#ifndef NNUE_FEATURES_ADVANCED_PAWN_PAIRS_INCLUDED
#define NNUE_FEATURES_ADVANCED_PAWN_PAIRS_INCLUDED

#include "full_threats.h"

namespace Stockfish::Eval::NNUE::Features {

// Actual sixth/seventh-rank pawns paired with knights, bishops, rooks and queens.
// Enabled with at most five non-pawns, counting both kings.
class AdvancedPawnPairs {
   public:
    static constexpr u32       HashValue           = 0x8d36b4e1u;
    static constexpr IndexType Dimensions          = 2 * 16 * 8 * 64;
    static constexpr IndexType MaxActiveDimensions = 16 * 3;
    using IndexList                                = ValueList<u16, MaxActiveDimensions>;
    using Snapshot                                 = DirtyAdvancedPawnPairs::Snapshot;

    static Snapshot  snapshot(const Position& pos);
    static IndexType make_index(
      Color perspective, Color pawnColor, Square pawn, Piece partner, Square square, Square ksq);
    static void append_active_indices(Color perspective, const Position& pos, IndexList& active);
    static void append_changed_indices(Color                         perspective,
                                       Square                        ksq,
                                       const DirtyAdvancedPawnPairs& diff,
                                       IndexList&                    removed,
                                       IndexList&                    added);
};

}  // namespace Stockfish::Eval::NNUE::Features
#endif
