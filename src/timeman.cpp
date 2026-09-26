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

#include "timeman.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "position.h"
#include "search.h"
#include "types.h"
#include "ucioption.h"

namespace Stockfish {

TimePoint TimeManagement::optimum() const { return optimumTime; }
TimePoint TimeManagement::maximum() const { return maximumTime; }

void TimeManagement::clear() {
    availableNodes    = -1;  // When in 'nodes as time' mode
    previousMovesToGo = 0;
}

void TimeManagement::advance_nodes_time(i64 nodes) {
    assert(useNodesTime);
    availableNodes = std::max(i64(0), availableNodes - nodes);
}

// Called at the beginning of the search and calculates
// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)
void TimeManagement::init(Search::LimitsType& limits,
                          const Position&     pos,
                          const OptionsMap&   options,
                          [[maybe_unused]] double& initialGameSec,
                          u64                 totalGameNodes,
                          TimePoint           totalGameTimeMs) {
    TimePoint npmsec = TimePoint(options["nodestime"]);

    Color us  = pos.side_to_move();
    int   ply = pos.game_ply();

    // If we have no time, we don't need to fully initialize TM.
    // startTime is used by movetime and useNodesTime is used in elapsed calls.
    startTime    = limits.startTime;
    useNodesTime = npmsec != 0;

    if (useNodesTime)
        limits.movetime *= npmsec;

    if (limits.time[us] == 0)
    {
        optimumTime = maximumTime = NoBound;
        return;
    }

    TimePoint moveOverhead = TimePoint(options["Move Overhead"]);

    // If we have to play in 'nodes as time' mode, then convert from time
    // to nodes, and use resulting values in time management formulas.
    // WARNING: to avoid time losses, the given npmsec (nodes per millisecond)
    // must be much lower than the real engine speed.
    if (useNodesTime)
    {
        if (availableNodes == -1)  // Only once at game start
        {
            // First time limit includes increment (both are in milliseconds)
            availableNodes = npmsec * limits.time[us];
            cyclicBudget   = npmsec * (limits.time[us] - limits.inc[us]);
        }
        else if (limits.movestogo > 0 && limits.movestogo > previousMovesToGo && cyclicBudget > 0)
            availableNodes += cyclicBudget;

        previousMovesToGo = limits.movestogo;

        // Convert from milliseconds to nodes
        limits.time[us] = TimePoint(availableNodes);
        limits.inc[us] *= npmsec;
        limits.npmsec = npmsec;
        moveOverhead *= npmsec;
    }

    // These numbers are used where multiplications, divisions,
    // or comparisons with constants are involved.
    const i64 scaleFactor = useNodesTime ? npmsec : 1;

    // Sudden death (no moves to go specified)
    if (limits.movestogo == 0)
    {
        // Net per-move increment cash flow after accounting for communication/execution overhead.
        // Allowed to be negative in sudden death / low increment formats, naturally deducting overhead.
        double effectiveInc = double(limits.inc[us] - moveOverhead);

        // Effective worker computation speed (nodes per second)
        int          threadsCount = std::max(1, int(options["Threads"]));
        const double referenceNPS = 628'000.0;
        double       effectiveNPS = referenceNPS * std::pow(threadsCount, 0.85);

        if (useNodesTime)
            effectiveNPS = double(npmsec) * 1000.0;
        else if (totalGameTimeMs >= 100)
            effectiveNPS = (double(totalGameNodes) * 1000.0) / double(totalGameTimeMs);

        // 1. Physically grounded move horizon:
        // A move costs communication latency (moveOverhead) PLUS the minimum computational effort
        // to complete root moves evaluation and initial tactical checks (~2500 nodes).
        double minSearchMs = (2500.0 * 1000.0) / effectiveNPS;
        double netDrain    = minSearchMs - effectiveInc;

        double M_pieces = std::max(8.0, 10.0 + 2.0 * pos.count<ALL_PIECES>());
        double M        = M_pieces;
        double sdScale  = 1.0;

        if (netDrain > 0.0)
        {
            double turnoverThreshold = M_pieces * netDrain;
            if (double(limits.time[us]) < turnoverThreshold && turnoverThreshold > 0.0)
            {
                double u = std::clamp(double(limits.time[us]) / turnoverThreshold, 0.0, 1.0);
                constexpr double c_sd = 0.50;
                sdScale = (u * (1.0 + c_sd)) / (u + c_sd);
                M       = 2.0 + (M_pieces - 2.0) * u;
            }
        }

        // Remaining game duration across our physically anchored horizon M with distributed liquidity balance (delta = 0.07).
        // Balances liquid bank capital against future increment cash flow:
        // (1 + delta) * T + (1 - delta) * I = (T + I) + delta * (T - I)
        double totalExpectedMs = double(limits.time[us]) + (M - 1.0) * effectiveInc;
        constexpr double delta = 0.07;
        double remainingGameMs =
          (totalExpectedMs + delta * (double(limits.time[us]) - (M - 1.0) * effectiveInc)) / double(scaleFactor);
        double remainingGameSec = std::max(0.01, remainingGameMs / 1000.0);

        double effectiveGameSec = remainingGameSec * (effectiveNPS / referenceNPS);
        double tauRaw           = std::log10(std::max(1.0, effectiveGameSec));
        double tauRawActual     = std::log10(effectiveGameSec);

        // Front-loading intensity tau via wide-scale sigmoid (k = 1.10, x0 = 1.35) into 2nd-order polynomial:
        // P(s) = c1 * s + (1.0 - c1) * s^2 where s in [0, 1]
        // Provides a wide linear regime spanning 3.5 decades: lifts front-loading in 2s-10s Sudden Death
        // while preserving headroom and smooth progression for STC, LTC, and Classical TCs without truncation.
        constexpr double deltaTau = 3.50;
        constexpr double k        = 1.10;
        constexpr double x0       = 1.35;
        constexpr double c1       = 0.25;

        double s = 0.0;
        if (tauRawActual > -2.0)
            s = 1.0 / (1.0 + std::exp(-k * (tauRawActual - x0)));

        double p_s = c1 * s + (1.0 - c1) * (s * s);
        double tau = 1.0 + deltaTau * p_s;

        // 2. Time Bank & Nominal Draw with Discrete Renewal Horizon
        // Time bank deducts safety reserve and the current move's incoming increment
        TimePoint safetyReserve = moveOverhead * 10;
        TimePoint timeBank =
          std::max(TimePoint(0), limits.time[us] - safetyReserve - TimePoint(std::max(0.0, effectiveInc)));

        // 3. Multiplicative bank factor without matFrac (dynamic horizon M governs game phase)
        double bankDraw = double(timeBank) * (tau / (M + tau));

        // Decrease time bank draw if behind in time.
        // This is skipped if the nodestime option is used because we can't calculate
        // the opponent nodes budget in a deterministic way.
        // We apply timeAdvantage strictly to bankDraw to conserve bank without
        // starving the engine below its incoming per-move increment cash flow.
        if (!useNodesTime)
        {
            double timeAdvantage =
              (double(limits.time[us]) - double(limits.time[~us])) / (1.0 + double(limits.time[us]) + double(limits.time[~us]));
            bankDraw *= (1.0 + 0.3 * std::min(timeAdvantage, 0.0));
        }

        // 4. Base move budget combining time bank draw and net increment cash flow,
        // gently scaled down in low-clock sudden death to avoid sudden cash flow wipeout.
        double baseMoveBudget = std::max(0.0, (bankDraw + effectiveInc) * sdScale);

        // 5. Early ply discount via smooth hyperbolic tangent (tanh) asymptotic saturation at Move 25 (Ply 50)
        // Opening discount is completely decoupled / invisible from tau, with fixed baseline floor w0 = 0.40
        constexpr double w0      = 0.40;
        double w_ply             = w0 + (1.0 - w0) * std::tanh(double(ply) / 20.0);
        TimePoint nominalOptimum = std::max(TimePoint(1), TimePoint(baseMoveBudget * w_ply));

        // 6. Dynamic maxScale ceiling
        double maxConstant       = std::max(3.3744 + 3.0608 * tauRaw, 3.1441);
        double maxScale          = std::min(6.873, maxConstant + ply / 12.352);
        TimePoint nominalMaximum = std::max(nominalOptimum, TimePoint(nominalOptimum * maxScale));

        // 7. Protect search extensions from blowing up clock when remaining time is critical
        optimumTime = nominalOptimum;
        maximumTime = nominalMaximum;
        double protectThreshold = 2.0 * double(nominalMaximum);
        if (protectThreshold > 0.0 && double(limits.time[us]) < protectThreshold)
        {
            constexpr double c = 0.25;
            double u     = std::clamp(double(limits.time[us]) / protectThreshold, 0.0, 1.0);
            double scale = (u * (1.0 + c)) / (u + c);
            optimumTime  = std::max(TimePoint(1), TimePoint(double(nominalOptimum) * scale));
            maximumTime  = std::max(TimePoint(1), TimePoint(double(nominalMaximum) * scale));
        }

        // 8. Hard safety caps
        TimePoint maxClockCap = std::max(TimePoint(1), TimePoint(0.8097 * limits.time[us] - moveOverhead));
        optimumTime           = std::min(optimumTime, maxClockCap);
        maximumTime           = std::max(optimumTime, std::min(maxClockCap, maximumTime));
    }

    // Cyclic time controls (x moves in y seconds + z increment)
    else
    {
        int mtg = std::min(limits.movestogo, 50);

        // Make sure timeLeft is > 0 since we may use it as a divisor
        TimePoint timeLeft = std::max(TimePoint(1), limits.time[us] + limits.inc[us] * (mtg - 1)
                                                      - moveOverhead * (2 + mtg));

        double optScale = std::min((0.88 + ply / 116.4) / mtg, 0.88 * limits.time[us] / timeLeft);
        double maxScale = 1.3 + 0.11 * mtg;

        // Decrease time usage if behind in time.
        // This is skipped in two cases:
        // - if the nodestime option is used we can't calculate the opponent nodes budget in a deterministic way.
        // - if we use a cyclic time management (like 40/10) calculating time advantage for the last move (movestogo = 1)
        //   can be vastly off, because if the opponent had done his last move before us his time budget includes already
        //   the next cycle time increment but our not. This leads to an unnecessary big decrease in time usage which favors blunders.
        // Warning: don't remove these conditions.
        if (!useNodesTime && limits.movestogo != 1)
        {
            double timeAdvantage =
              (limits.time[us] - limits.time[~us]) / (1.0 + limits.time[us] + limits.time[~us]);
            optScale *= 1 + 0.9 * std::min(timeAdvantage, 0.0);
        }

        optimumTime = TimePoint(std::max(1.0, optScale * timeLeft));
        maximumTime =
          TimePoint(std::max(double(optimumTime), std::min(0.8097 * limits.time[us] - moveOverhead,
                                                           maxScale * optimumTime)));
    }

    if (options["Ponder"])
        optimumTime += optimumTime / 4;
}

}  // namespace Stockfish
