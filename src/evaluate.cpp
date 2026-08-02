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

namespace ScaleParams {
    // Normalization: controls how quickly the sigmoid flattens for alignment.
    constexpr int SeNormDiv   = 1024;
    constexpr int NnueNormDiv = 1024;

    // Alignment: tunable multipliers.
    // OptAlignMult = 64 matches old OptAlignDiv = 256.
    // NnueAlignMult = 32 matches old NnueAlignDiv = 512.
    constexpr int OptAlignMult  = 64;
    constexpr int NnueAlignMult = 32;
    constexpr int AlignBaseDiv  = 16384; // 2^14 (DO NOT TUNE)

    // Material interpolation
    constexpr int MaxMaterial = 32768;
    constexpr int MatBase     = 7191;

    // Confidence scaling
    constexpr int SeConfBase  = 512;
    constexpr int SeConfMax   = 3000; // Hard cap to prevent extreme material blow-ups

    // Final evaluation scaling
    // 839 / 67108864 is approximately 1 / 79986.7
    constexpr int OptScaleMult = 839;
    constexpr int OptScaleDiv  = 67108864; // 2^26 (DO NOT TUNE)
    constexpr int Rule50Div    = 199;
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

// Applies search-dependent scaling (optimism and rule50) to the raw NNUE eval
Value Eval::scale_evaluation(Value nnue, int optimism, const Position& pos) {
    int se = simple_eval(pos);
    int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();

    // 1. Normalize SE and NNUE for ALIGNMENT only
    int se_norm   = (se * 512) / (std::abs(se) + ScaleParams::SeNormDiv);
    int nnue_norm = (nnue * 512) / (std::abs(nnue) + ScaleParams::NnueNormDiv);

    // 2. Measure Alignment
    int opt_align  = optimism * se_norm;
    int nnue_align = nnue_norm * se_norm;

    int alignment = (opt_align * ScaleParams::OptAlignMult +
                     nnue_align * ScaleParams::NnueAlignMult) / ScaleParams::AlignBaseDiv;

    // 3. Create a blending weight [0, 1024]
    int weight = 512 - std::clamp(alignment, -512, 512);

    // 4. Smoothly interpolate between the two material philosophies.
    int inverted_material = std::max(0, ScaleParams::MaxMaterial - material);

    int effective_material = (weight * material + (1024 - weight) * inverted_material) / 1024;
    int mat_multiplier = ScaleParams::MatBase + effective_material;

    // 5. Apply optimism scaling SAFELY.
    int se_confidence = ScaleParams::SeConfBase + std::min(std::abs(se), ScaleParams::SeConfMax);

    // We retain i64 casts here to prevent intermediate overflow before the division
    optimism = (optimism * i64(se_confidence) * i64(mat_multiplier)) / 512;

    // 6. Direct addition of scaled optimism to NNUE.
    // Cast to i64 protects against overflow from the OptScaleMult multiplication.
    int v = nnue + (optimism * i64(ScaleParams::OptScaleMult)) / ScaleParams::OptScaleDiv;

    // Damp down the evaluation linearly when shuffling
    v -= v * pos.rule50_count() / ScaleParams::Rule50Div;

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
