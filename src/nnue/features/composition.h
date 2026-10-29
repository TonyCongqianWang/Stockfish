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

// Composition-conditioned material. Relative classes: own P,N,B,R,Q, enemy P,N,B,R,Q.
#ifndef NNUE_FEATURES_COMPOSITION_H_INCLUDED
#define NNUE_FEATURES_COMPOSITION_H_INCLUDED

#include <array>
#include "../../position.h"

namespace Stockfish::Eval::NNUE::Features {
struct Composition {
#ifdef NNUE_COMPOSITION_TEST
    // Test-only counters distinguish state deltas from cold/stale Finny entries.
    static inline usize incrementalUpdates                = 0;
    static inline usize cacheUpdates                      = 0;
    static inline usize materialCacheHits                 = 0;
    static inline usize materialCacheMisses               = 0;
    static inline usize materialCacheCountReconciliations = 0;
    static inline usize materialCacheCollisions           = 0;
#endif
    static constexpr u32 HashValue  = 0xC04D1024u;
    static constexpr int Dimensions = 5120;
    using Counts = std::array<u8, 10>;  // absolute white classes, then black classes

    static Counts counts(const Position& pos) {
        Counts result{};
        for (Color c : {WHITE, BLACK})
            for (int t = PAWN; t <= QUEEN; ++t)
                result[5 * c + t - PAWN] = u8(popcount(pos.pieces(c, PieceType(t))));
        return result;
    }
    static int bucket(const Counts& counts, Color perspective) {
        int b = 0;
        for (int t = 0; t < 10; ++t)
            if (counts[(t + 5 * perspective) % 10])
                b |= 1 << t;
        return b;
    }
    // Bucket-major packed rows; absent classes have no row.
    static constexpr auto Offsets = [] {
        std::array<int, 1024> result{};
        int                   n = 0;
        for (int b = 0; b < 1024; ++b)
        {
            result[b] = n;
            for (int t = 0; t < 10; ++t)
                n += (b >> t) & 1;
        }
        return result;
    }();
    static bool changes(const DirtyPiece& d) {
        return d.to == SQ_NONE || (d.remove_sq != SQ_NONE && d.add_sq == SQ_NONE);
    }
    static Counts changed(Counts counts, const DirtyPiece& d, int sign) {
        auto add = [&](Piece p, int delta) {
            if (type_of(p) != KING)
                counts[5 * color_of(p) + type_of(p) - PAWN] += delta;
        };
        if (d.to == SQ_NONE)
            add(d.pc, -sign);
        if (d.remove_sq != SQ_NONE)
            add(d.remove_pc, -sign);
        if (d.add_sq != SQ_NONE)
            add(d.add_pc, sign);
        return counts;
    }
};
}
#endif
