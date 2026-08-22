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

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

int Eval::simple_eval(const Position& pos) {
    const Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c)) + pos.non_pawn_material(c)
         - pos.non_pawn_material(~c);
}

Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches) {

    assert(!pos.checkers());
    auto [psqt, positional] = network.evaluate(pos, accumulators, caches);
    return psqt + positional;
}

// Applies search-dependent scaling (root score and rule50) to the raw NNUE eval
Value Eval::scale_evaluation(Value nnue, int rootScore, const Position& pos) {
    int se = Eval::simple_eval(pos);

    // scale se inversely with material (roughly triple se when material is 0)
    int material = 521 * pos.count<PAWN>() + pos.non_pawn_material();
    se = i64(se) * 30000 / (material + 10000);

    // Normalize the raw evaluations to [-1024, 1024] to measure their correlation.
    int se_norm   = i64(se) * 1024 / (std::abs(se) + 512);
    int nnue_norm = i64(nnue) * 1024 / (std::abs(nnue) + 512);
    // When NNUE and material agree, the position is straightforward; otherwise, it
    // involves complex compensation. In a representative sample, raw_alignment
    // averages -1 or so, i.e. it is well-centered in [-2048, 2048].
    int raw_alignment = (se_norm * nnue_norm) / 2048;

    // Normalize the slow root-score EMA to [-1024, 1024], mirroring se_norm/nnue_norm.
    // Positive rs_norm means we are winning from the root player's (stm's) perspective.
    int rs_norm = i64(rootScore) * 128 / (std::abs(rootScore) + 95);

    // When winning, we favor easy positions with higher alignment, and vice versa.
    // As raw_alignment is centered, overall eval scale is preserved
    int base_eval = (i64(nnue) * 76504105 + 1024 * (nnue * raw_alignment) + 4096 * (rs_norm * raw_alignment)) / 67108864;

    int v = base_eval;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 189;

    // Guarantee that the evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Network& network) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(network);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, network, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    auto [psqt, positional] = network.evaluate(pos, *accumulators, *caches);
    Value nnue              = psqt + positional;
    Value s_v  = Eval::scale_evaluation(nnue, VALUE_ZERO, pos); // requires stm perspective

    ss << "NNUE evaluation          " << nnue << " (side to move, internal units)\n";

    nnue = pos.side_to_move() == WHITE ? nnue : -nnue;
    s_v  = pos.side_to_move() == WHITE ? s_v : -s_v;

    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(nnue, pos) << " (white side, pawns)\n";
    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(s_v, pos) << " (white side, pawns)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
