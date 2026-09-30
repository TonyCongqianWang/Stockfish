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

static int pawn_value(int npm) {
    return 5750000 / (11034 + npm);
}

static int simple_eval(const Position& pos, int pv) {
    const Color c = pos.side_to_move();
    return pv * (pos.count<PAWN>(c) - pos.count<PAWN>(~c)) + pos.non_pawn_material(c)
         - pos.non_pawn_material(~c);
}

// Algebraic WDL margin (Win - Loss) normalized to [-1024, 1024]
static int wdl_margin(int v, int a, int b) {
    int tp = (v + a) * 1024 / (std::abs(v + a) + b);
    int tn = (v - a) * 1024 / (std::abs(v - a) + b);
    return (tp + tn) / 2;
}

Value scale_evaluation(Value nnue, int optimism, const Position& pos);

Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());
    Value nnue = network.evaluate(pos, accumulators, caches);
    return scale_evaluation(nnue, optimism, pos);
}

// Applies search-dependent scaling (optimism and rule50) to the raw NNUE eval
Value scale_evaluation(Value nnue, int optimism, const Position& pos) {
    int npm = pos.non_pawn_material();
    int pv  = pawn_value(npm);
    int se  = simple_eval(pos, pv);

    int material = pv * pos.count<PAWN>() + npm;

    // Scale NNUE into search evaluation space (matches the UCI WDL domain)
    Value nnue_v = nnue * i64(90649 + material) / 90649;

    // Exact inverse scaling for simple eval
    int se_scaled = int(se * i64(90649) / (90649 + material));

    // Algebraic WDL parameters in search space
    constexpr int a_nnue = 310;
    int a_se = 310 + (pos.count<PAWN>() == 0) * (7 * std::max(0, 8000 - npm)) / 80;
    constexpr int b = 70;

    int se_margin   = wdl_margin(se_scaled, a_se, b);
    int nnue_margin = wdl_margin(nnue_v, a_nnue, b);

    // Alignment measures directional concordance; margin distance measures dynamic complexity
    int alignment  = (se_margin * nnue_margin) / 512;
    int complexity = std::abs(se_margin - nnue_margin) - 256;

    int se_adjust = alignment + complexity;

    // When winning, we favor easy positions and dynamic tension over flat draws, and vice versa
    int v = nnue_v + (nnue_v * se_adjust) / 65536 + (optimism * se_adjust) / 16384;

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

    Value nnue = network.evaluate(pos, *accumulators, *caches);
    Value s_v  = scale_evaluation(nnue, VALUE_ZERO, pos);  // requires stm perspective

    ss << "NNUE evaluation          " << nnue << " (side to move, internal units)\n";

    nnue = pos.side_to_move() == WHITE ? nnue : -nnue;
    s_v  = pos.side_to_move() == WHITE ? s_v : -s_v;

    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(nnue, pos) << " (white side)\n";
    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(s_v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
