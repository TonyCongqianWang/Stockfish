/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)
  SPDX-License-Identifier: GPL-3.0-or-later
*/

#ifndef NNUE_FEATURES_ENDGAME_H_INCLUDED
#define NNUE_FEATURES_ENDGAME_H_INCLUDED

#include <array>
#include "../../position.h"

namespace Stockfish::Eval::NNUE::Features {

// Independent inventories for each piece color; mixed inventories are silent.
// Pawns: none/N/B/R/Q/NR/BR/NB. Pieces: N/B/R/Q/NR/BR/NB, with/without pawns.
struct Endgame {
#ifdef NNUE_ENDGAME_TEST
    static inline usize rowUpdates      = 0;
    static inline usize reconciliations = 0;
#endif
    static constexpr u32 HashValue  = 0xE8D61408u;
    static constexpr int Dimensions = 3584;
    using Board                     = std::array<Bitboard, 5>;
    using Snapshot                  = std::array<Board, COLOR_NB>;

    static Snapshot snapshot(const Position& pos) {
        Snapshot result;
        for (Color c : {WHITE, BLACK})
        {
            for (int t = PAWN; t <= QUEEN; ++t)
                result[c][t - PAWN] = pos.pieces(c, PieceType(t));
            if (category(result[c]) < 0)
                result[c].fill(0);
        }
        return result;
    }

    static int category(const Board& b) {
        int mask = 0;
        for (int t = 1; t < 5; ++t)
            mask |= bool(b[t]) << (t - 1);
        switch (mask)
        {
        case 0 :
            return b[0] ? 0 : -1;
        case 1 :
            return 1;
        case 2 :
            return 2;
        case 4 :
            return 3;
        case 8 :
            return 4;
        case 5 :
            return popcount(b[1]) == 1 && popcount(b[3]) == 1 ? 5 : -1;
        case 6 :
            return popcount(b[2]) == 1 && popcount(b[3]) == 1 ? 6 : -1;
        case 3 :
            return popcount(b[1]) == 1 && popcount(b[2]) == 1 ? 7 : -1;
        default :
            return -1;
        }
    }

    static int orientation(Color perspective, Square king) {
        return (perspective == BLACK ? 56 : 0) ^ (file_of(king) < FILE_E ? 7 : 0);
    }

    // Pack only piece types that can occur in each category; no king rows.
    static int make_index(int category, bool pawns, int type, bool enemy, Square sq, int flip) {
        constexpr int offsets[8] = {0, 0, 1, 2, 3, 4, 6, 8};
        const int     plane      = type == 0 ? category
                                             : 8 + 10 * !pawns + offsets[category]
                                        + (category >= 5 && type == (category == 7 ? 2 : 3));
        return plane * 128 + int(enemy) * 64 + (int(sq) ^ flip);
    }
};

}  // namespace Stockfish::Eval::NNUE::Features
#endif
