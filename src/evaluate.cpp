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
    return psqt + positional;
}

namespace ScaleParams {
    // Normalization parameters for evaluation alignment.
    constexpr int SeNormDiv   = 1024;
    constexpr int NnueNormDiv = 1024;

    // Alignment shifts the raw [-2048, 2048] correlation into a strictly positive [0, 4096] multiplier.
    // Empirically, over representative game positions: raw_alignment mean is very close to 0 (-0.96).
    // So alignment (with offset) is typically centered near 2048.
    constexpr int AlignOffset   = 2048;

    // Tunable weights for blending optimism based on position difficulty.
    //
    // Optimism adjustment: opt += opt * alignment * OptAlignMult / OptAlignDiv
    //   At mean alignment≈2047: factor = 2047*4/16384 ≈ 1.50x boost at typical position.
    //   This matches master's (1 + complexity/474) at typical complexity ~234 (giving 1.49x).
    constexpr int OptAlignMult  = 4;
    constexpr int OptAlignDiv   = 16384;

    // NNUE adjustment: nnue -= nnue * alignment * NnueAlignMult / NnueAlignDiv
    //   Master reduces NNUE by ~nnueComplexity/19163 ≈ 1.22% at typical complexity~234.
    //   With alignment≈2047, NnueAlignDiv is chosen as a power of 2 for shift optimization.
    //   Target: 2047 * 1 / 131072 ≈ 1.56% damping, matching master's average damping scale.
    constexpr int NnueAlignMult = 1;
    constexpr int NnueAlignDiv  = 131072;

    // Base evaluation scaling constants, matched to the updated pending master parameters:
    //   material = 521*pawns + npm  (was 534)
    //   EvalBaseDiv = 90649         (was 77871)
    //   OptWeight = 7674/90649      (matches master's 7674 optimism contribution)
    constexpr int EvalBaseDiv  = 90649;
    constexpr int OptWeight    = 7674;
    constexpr int PawnWeight   = 521;
    constexpr int Rule50Div    = 189;
}

// Applies search-dependent scaling (optimism and rule50) to the raw NNUE eval
Value Eval::scale_evaluation(Value nnue, int optimism, const Position& pos) {
    int se = simple_eval(pos);

    // 1. Normalize the static evaluations to a bounded [-1024, 1024] range.
    // This stabilizes the inputs before we measure their correlation.
    int se_norm   = (se * 1024) / (std::abs(se) + ScaleParams::SeNormDiv);
    int nnue_norm = (nnue * 1024) / (std::abs(nnue) + ScaleParams::NnueNormDiv);

    // 2. Measure positional difficulty (Alignment).
    // When NNUE and simple_eval (material) agree, the position is generally straightforward ("easy").
    // When they disagree, the position involves complex compensation ("hard").
    // The raw product is shifted by AlignOffset to guarantee a strictly positive modifier,
    // yielding high values (~4096) for easy positions and low values (~0) for hard ones.
    int raw_alignment = (se_norm * nnue_norm) / 512;
    int alignment = raw_alignment + ScaleParams::AlignOffset;

    // 3. Blend optimism and NNUE using the difficulty modifier.
    // We favor easily convertible positions by heavily boosting optimism when alignment is high.
    // Conversely, in complex/hard positions, the optimism boost is minimized.
    // To maintain overall evaluation scale, the static NNUE score is dampened proportionally.
    optimism += (optimism * alignment * ScaleParams::OptAlignMult) / ScaleParams::OptAlignDiv;
    nnue     -= (i64(nnue) * alignment * ScaleParams::NnueAlignMult) / ScaleParams::NnueAlignDiv;

    // 4. Combine into a fixed-ratio base evaluation.
    int scaled_optimism = (optimism * ScaleParams::OptWeight) / ScaleParams::EvalBaseDiv;
    int base_eval       = nnue + scaled_optimism;

    // 5. Scale the combined evaluation by material volume.
    // Higher material on the board amplifies the final evaluation magnitude.
    int material = ScaleParams::PawnWeight * pos.count<PAWN>() + pos.non_pawn_material();
    int v = (base_eval * i64(ScaleParams::EvalBaseDiv + material)) / ScaleParams::EvalBaseDiv;

    // Damp down the evaluation linearly when shuffling (approaching the 50-move rule limit).
    v -= v * pos.rule50_count() / ScaleParams::Rule50Div;

    // Guarantee the evaluation stays within standard engine bounds, avoiding tablebase score collisions.
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

    Value v = evaluate(network, pos, *accumulators, *caches);
    ss << "NNUE evaluation          " << v << " (side to move, internal units)\n";
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(pos.side_to_move() == WHITE ? v : -v, pos)
       << " (white side)\n";

    // scale_evaluation expects STM perspective (same as evaluate()), so call it before sign flip
    v = Eval::scale_evaluation(v, VALUE_ZERO, pos);
    v = pos.side_to_move() == WHITE ? v : -v;

    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
