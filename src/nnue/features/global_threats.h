/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef NNUE_FEATURES_GLOBAL_THREATS_INCLUDED
#define NNUE_FEATURES_GLOBAL_THREATS_INCLUDED

#include "../../misc.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE::Features {

// All relations use the receiving king's horizontal orientation. Equal-type
// reciprocal contacts are represented once, in descending physical-square order.
class GlobalSupports {
   public:
    static constexpr u32       OwnHash = 0x4f535031u, EnemyHash = 0x45535031u;
    static constexpr IndexType Dimensions = 14952;
    using IndexList                       = ValueList<u16, 256>;

    static IndexType make_index(Color, Piece, Square, Square, Piece, Square = SQ_NONE);
    static void      append_active_indices(Color, bool enemy, const Position&, IndexList&);
    static void      append_changed_indices(
      Color, bool enemy, Square, const DirtyThreats&, IndexList& removed, IndexList& added);
};

class GlobalAttacks {
   public:
    static constexpr u32       HashValue           = 0x47415432u;
    static constexpr IndexType Dimensions          = 29904;
    static constexpr IndexType MaxActiveDimensions = 256;
    using IndexList                                = ValueList<u16, MaxActiveDimensions>;
    using DiffType                                 = DirtyThreats;

    static IndexType make_index(
      Color perspective, Piece attacker, Square from, Square to, Piece attacked, Square = SQ_NONE);
    static void append_active_indices(Color, const Position&, IndexList&);
    static void append_changed_indices(Color,
                                       Square,
                                       const DiffType&,
                                       IndexList& removed,
                                       IndexList& added,
                                       const ThreatWeightType* = nullptr,
                                       IndexType               = 0);
    static void append_changed_indices_both(Square,
                                            Square,
                                            const DiffType&,
                                            IndexList& white_removed,
                                            IndexList& white_added,
                                            IndexList& black_removed,
                                            IndexList& black_added,
                                            const ThreatWeightType* = nullptr,
                                            IndexType               = 0);
};

}  // namespace Stockfish::Eval::NNUE::Features
#endif
