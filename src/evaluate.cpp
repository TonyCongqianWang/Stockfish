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

namespace Stockfish {

static Value shuffle_dampening(const Position& pos, Value v) {
    // Do not tune sd_r_lut[0].
    constexpr std::array<uint16_t, 128> sd_r_lut = {
        0, 73, 147, 220, 295, 369, 445, 520,
        597, 674, 752, 830, 909, 989, 1070, 1152,
        1234, 1317, 1401, 1486, 1571, 1657, 1743, 1829,
        1916, 2002, 2089, 2176, 2262, 2348, 2434, 2519,
        2604, 2688, 2771, 2854, 2935, 3016, 3095, 3173,
        3250, 3325, 3399, 3471, 3542, 3611, 3677, 3742,
        3805, 3866, 3925, 3981, 4037, 4090, 4142, 4193,
        4243, 4292, 4341, 4388, 4436, 4483, 4530, 4577,
        4624, 4672, 4720, 4769, 4818, 4869, 4921, 4974,
        5029, 5085, 5143, 5203, 5266, 5330, 5397, 5467,
        5539, 5614, 5692, 5774, 5859, 5948, 6040, 6136,
        6236, 6340, 6449, 6561, 6678, 6798, 6921, 7049,
        7179, 7314, 7451, 7591, 7735, 7881, 8031, 8183,
        8337, 8495, 8654, 8816, 8980, 9147, 9315, 9485,
        9657, 9830, 10005, 10182, 10360, 10539, 10720, 10901,
        11083, 11267, 11451, 11635, 11820, 12006, 12192, 12378
    };
    int move_count = std::min(static_cast<int>(pos.rule50_count()), 127);
    int r = sd_r_lut[move_count];
    // Branchless multiplier:
    // If pieces <= 6, condition is 1 -> 128 + 25 = 153
    // If pieces > 6,  condition is 0 -> 128 + 0  = 128
    // Divisors (and 128 multiplier) should not be tuned.
    r = (r * (128 + 25 * (pos.pieces() <= 6))) / 128;
    v -= static_cast<int64_t>(v) * r / 16384;
    return v;
}

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());

    auto [psqt, positional] = network.evaluate(pos, accumulators, caches);

    Value nnue = psqt + positional;

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = std::abs(psqt - positional);
    optimism += optimism * nnueComplexity / 476;
    nnue -= nnue * nnueComplexity / 18236;

    int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
    int v        = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;

    v = shuffle_dampening(pos, v);

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
