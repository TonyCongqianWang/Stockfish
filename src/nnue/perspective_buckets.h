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

// Handcrafted perspective routing. No learned routing parameters.
#ifndef NNUE_PERSPECTIVE_BUCKETS_H_INCLUDED
#define NNUE_PERSPECTIVE_BUCKETS_H_INCLUDED
#include <algorithm>
#include <array>
#include "../position.h"
namespace Stockfish::Eval::NNUE {
constexpr int LiveExperts     = 85;
constexpr int FallbackExpert  = LiveExperts;
constexpr int RoleExperts     = LiveExperts + 1;
constexpr int InvalidBucket   = 255;
constexpr u32 PerspectiveHash = 0x48503835u;
// heavy: Q, R, else; officers: <2, >=2; pawns: both, one, none.
inline constexpr std::array<u8, 144> NominalToCompact = {
  255, 255, 0,   255, 255, 255, 255, 255, 1,   255, 255, 255, 255, 2,   3,   255, 255, 255,
  4,   5,   255, 255, 6,   7,   8,   9,   255, 255, 10,  11,  12,  13,  255, 255, 14,  15,
  16,  17,  255, 18,  19,  20,  21,  22,  255, 23,  24,  25,  26,  27,  255, 28,  29,  30,
  31,  32,  255, 33,  34,  35,  36,  37,  255, 38,  39,  40,  41,  42,  255, 43,  44,  45,
  46,  47,  255, 48,  49,  50,  51,  52,  255, 53,  54,  55,  56,  57,  255, 58,  59,  60,
  255, 255, 255, 61,  62,  63,  255, 255, 255, 64,  65,  66,  255, 255, 255, 67,  68,  69,
  255, 255, 255, 70,  71,  72,  255, 255, 255, 73,  74,  75,  255, 255, 255, 76,  77,  78,
  255, 255, 255, 79,  80,  81,  255, 255, 255, 82,  83,  84,  255, 255, 255, 255, 255, 255,
};
inline int material_bucket(
  int pawns, int knights, int bishops, int rooks, int queens, bool leftPawns, bool rightPawns) {
    const int officers   = knights + bishops + rooks + queens;
    const int total      = pawns + officers;
    const int promotions = std::max(0, knights - 2) + std::max(0, bishops - 2)
                         + std::max(0, rooks - 2) + std::max(0, queens - 1);
    if (pawns > 8 || total > 15 || promotions > 8 - pawns)
        return InvalidBucket;
    const int layout = !pawns ? 2 : leftPawns && rightPawns ? 0 : 1;
    const int heavy  = queens ? 0 : rooks ? 1 : 2;
    return NominalToCompact[(((total / 2) * 3 + heavy) * 2 + (officers >= 2)) * 3 + layout];
}
inline int perspective_bucket(const Position& pos, Color p) {
    constexpr Bitboard Left  = FileABB | FileBBB | FileCBB | FileDBB;
    const Bitboard     pawns = pos.pieces(p, PAWN);
    return material_bucket(pos.count<PAWN>(p), pos.count<KNIGHT>(p), pos.count<BISHOP>(p),
                           pos.count<ROOK>(p), pos.count<QUEEN>(p), pawns & Left, pawns & ~Left);
}
struct PerspectiveRoute {
    int  bucket;
    int  expert;
    bool unsupported;
    bool unexposed;
};
inline PerspectiveRoute perspective_route(const Position& pos, Color p, const u8* mask) {
    const int  b           = perspective_bucket(pos, p);
    const bool unsupported = b == InvalidBucket;
    const bool unexposed   = !unsupported && !(mask[b / 8] & (1u << (b % 8)));
    return {b, unsupported || unexposed ? FallbackExpert : b, unsupported, unexposed};
}
}
#endif
