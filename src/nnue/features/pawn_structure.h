/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  SPDX-License-Identifier: GPL-3.0-or-later
*/
#ifndef NNUE_FEATURES_PAWN_STRUCTURE_INCLUDED
#define NNUE_FEATURES_PAWN_STRUCTURE_INCLUDED

#include "global_threats.h"

namespace Stockfish::Eval::NNUE::Features {
class PawnStructure {
   public:
    static constexpr u32       FrontierHash = 0x50465032u, FileHash = 0x50464332u;
    static constexpr IndexType FrontierDimensions = 7 * 49 * 49;
    static constexpr IndexType FileDimensions     = 8 * 60;
    static constexpr IndexType Dimensions         = FrontierDimensions + FileDimensions;
    static constexpr IndexType IndexBase          = GlobalAttacks::Dimensions;
    using IndexList                               = GlobalAttacks::IndexList;
    using DiffType                                = DirtyPawnStructure;

    static IndexType
    frontier_index(Color, int file, Bitboard white, Bitboard black, Square = SQ_NONE);
    static IndexType file_pair_index(Color, int file, int first, int second, Square = SQ_NONE);
    static void      append_active_indices(Color, const Position&, IndexList&);
    static void      append_changed_indices(Color,
                                            Square,
                                            const DiffType&,
                                            IndexList& removed,
                                            IndexList& added,
                                            const ThreatWeightType* = nullptr,
                                            IndexType               = 0);
    static void      append_changed_indices_both(Square,
                                                 Square,
                                                 const DiffType&,
                                                 IndexList& wr,
                                                 IndexList& wa,
                                                 IndexList& br,
                                                 IndexList& ba,
                                                 const ThreatWeightType* = nullptr,
                                                 IndexType               = 0);
};
static_assert(PawnStructure::IndexBase + PawnStructure::Dimensions <= 65536);
}  // namespace Stockfish::Eval::NNUE::Features
#endif
