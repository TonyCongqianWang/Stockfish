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

int PSQT_W=278648, OPTIMISM_W=39852;
int NNUE_MAX=3*QueenValue, NNUE_COMPLEXITY_MAX=QueenValue, OPTIMISM_MAX=272;
int NNUE_LEAKY=256, NNUE_COMPLEXITY_LEAKY=256, OPTIMISM_LEAKY=1024;
int OPTIMISM_0=54, OPTIMISM_1=463, OPTIMISM_2=7418;

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());
    const int material = PawnValue * pos.count<PAWN>() + pos.non_pawn_material();

    auto [psqt, positional] = network.evaluate(pos, accumulators, caches);

    Value nnue = psqt + positional;

    // Blend optimism with nnue, nnueComplexity, and material.
    int nnueComplexity = std::abs(positional);
    nnueComplexity = std::clamp(nnueComplexity, -NNUE_COMPLEXITY_MAX, NNUE_COMPLEXITY_MAX) + nnueComplexity / NNUE_COMPLEXITY_LEAKY;
    nnue                     = std::clamp(nnue, -NNUE_MAX, NNUE_MAX) + nnue / NNUE_LEAKY;
    optimism = (static_cast<std::int64_t>(optimism) * (OPTIMISM_0 + std::abs(nnue)) * (OPTIMISM_1 + nnueComplexity)
                * (OPTIMISM_2 + material))
             / 536870912ll;
    optimism = std::clamp(optimism, -OPTIMISM_MAX, OPTIMISM_MAX) + optimism / OPTIMISM_LEAKY;
    int v = (PSQT_W * static_cast<std::int64_t>(nnue) + OPTIMISM_W * static_cast<std::int64_t>(optimism))
          / 262144;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / 199;

    // Guarantee evaluation does not hit the tablebase range
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
    Value v                 = psqt + positional;
    ss << "NNUE evaluation          " << v << " (side to move, internal units)\n";
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(network, pos, *accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;

    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
