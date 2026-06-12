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

#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

#include "tune.h"

namespace Stockfish {

constexpr int SCALE_SHIFT = 14;
constexpr int QUAD_SHIFT  = 24; // Deeper virtual bit-depth for the quadratic precision

// All values are now beautifully aligned in the thousands for SPSA
int VAL_NNUE_LINEAR = 16094, VAL_OPT_LINEAR = 2582, VAL_OPT_QUAD = 5120;
TUNE(VAL_NNUE_LINEAR, VAL_OPT_LINEAR, VAL_OPT_QUAD)

Value Eval::scale_nnue_eval(Value nnue, const Position& pos, int optimism) {

    int nnueMagnitude = std::abs(nnue);
    int optSign       = (optimism > 0) - (optimism < 0);
    int nnueSign      = (nnue > 0) - (nnue < 0);

    // 3 operational arithmetic steps as requested, strictly following your formula:
    i64 optLinear  = i64(std::abs(optimism)) * VAL_OPT_LINEAR;
    i64 optScaled  = optLinear + ((i64(std::abs(optimism)) * nnueMagnitude * VAL_OPT_QUAD) >> QUAD_SHIFT);
    i64 nnueScaled = i64(nnueMagnitude) * VAL_NNUE_LINEAR;

    // Reapply signs symmetrically using power-of-two division
    int v = (nnueScaled * nnueSign + optScaled * optSign) >> SCALE_SHIFT;

    // linear shuffle dampening.
    v -= v * pos.rule50_count() / 199;

    // clamping to avoid TB values.
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Eval::EvaluateOutput Eval::evaluate(const Eval::NNUE::Network&     network,
                              const Position&                pos,
                              Eval::NNUE::AccumulatorStack&  accumulators,
                              Eval::NNUE::AccumulatorCaches& caches,
                              int                            optimism) {

    assert(!pos.checkers());

    auto [psqt, positional] = network.evaluate(pos, accumulators, caches);

    Value nnue = psqt + positional;

    return std::make_tuple(nnue, scale_nnue_eval(nnue, pos, optimism));
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
    Value v                 = psqt + positional;
    ss << "NNUE evaluation          " << v << " (side to move, internal units)\n";
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    auto [nnue, scaled] = evaluate(network, pos, *accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? scaled : -scaled;

    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
