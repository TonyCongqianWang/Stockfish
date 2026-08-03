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

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the side to move. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos) {
    const Color c = pos.side_to_move();
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c)) + pos.non_pawn_material(c)
         - pos.non_pawn_material(~c);
}

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches) {

    assert(!pos.checkers());
    auto [psqt, positional] = network.evaluate(pos, accumulators, caches);
    return 1200 * i64(psqt + positional) / 1024;
}

// Applies search-dependent scaling (optimism and rule50) to the raw NNUE eval
Value Eval::scale_evaluation(Value nnue, int optimism, const Position& pos) {
    int se = simple_eval(pos);
    int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();

    // 1. Measure Alignment (Are optimism and simple_eval pulling in the same direction?)
    // The divisor (256) dictates the "width" of the transition zone.
    // Higher divisor = smoother, slower transition.
    int alignment = (optimism * se) / 256;

    // 2. Create a blending weight [0, 1024]
    // Clamp to [-512, 512], then shift.
    // Highly Aligned (+512)      -> weight = 0
    // Highly Anti-aligned (-512) -> weight = 1024
    // Neutral (0)                -> weight = 512
    int weight = 512 - std::clamp(alignment, -512, 512);

    // 3. Smoothly interpolate between the two material philosophies.
    // M_max is an approximation of full board material (using a fast power of 2).
    constexpr int M_max = 32768;
    int inverted_material = std::max(0, M_max - material);

    // Lerp: As weight approaches 0, we favor the inverted material (rewarding trades)
    int effective_material = (weight * material + (1024 - weight) * inverted_material) / 1024;

    int mat_multiplier = 7191 + effective_material;

    optimism = (optimism * i64(512 + std::abs(se)) * i64(mat_multiplier)) / 80000 / 512;
    int v        = nnue + optimism;

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

    Value v                 = evaluate(network, pos, *accumulators, *caches);
    ss << "NNUE evaluation          " << v << " (side to move, internal units)\n";
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = Eval::scale_evaluation(v, VALUE_ZERO, pos);
    v = pos.side_to_move() == WHITE ? v : -v;

    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
