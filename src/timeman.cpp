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
    initialTimeLeft   = -1;
    mainThreadNodes   = 0;
    mainThreadTimeMs  = 0;
    benchmarkedMoves  = 0;
}

void TimeManagement::advance_nodes_time(i64 nodes) {
    assert(useNodesTime);
    availableNodes = std::max(i64(0), availableNodes - nodes);
}

void TimeManagement::update_benchmark(u64 nodes, TimePoint elapsed, const OptionsMap& options) {
    if (useNodesTime || int(options["MachineSpeedAdjustment"]) > 0)
        return;

    if (benchmarkedMoves < 5 && (benchmarkedMoves < 3 || mainThreadTimeMs < 200))
    {
        mainThreadNodes  += nodes;
        mainThreadTimeMs += elapsed;
        benchmarkedMoves++;
    }
}

// Called at the beginning of the search and calculates
// the bounds of time allowed for the current game ply. We currently support:
//      1) x basetime (+ z increment)
//      2) x moves in y seconds (+ z increment)
void TimeManagement::init(Search::LimitsType& limits,
                          const Position&     pos,
                          const OptionsMap&   options) {
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
        int    threadsCount         = std::max(1, int(options["Threads"]));
        double threadScalingFactor  = std::sqrt(threadsCount);

        const double referenceNPS          = 467'000.0;
        double       speedAdjustmentFactor = 1.0;

        int machineSpeedAdjustment = int(options["MachineSpeedAdjustment"]);
        if (machineSpeedAdjustment > 0)
            speedAdjustmentFactor = double(machineSpeedAdjustment) / 1000.0;
        else if (!useNodesTime && mainThreadTimeMs > 0 && mainThreadNodes > 0)
        {
            double mainNPS        = (double(mainThreadNodes) * 1000.0) / double(mainThreadTimeMs);
            speedAdjustmentFactor = mainNPS / referenceNPS;
        }

        double totalScale   = speedAdjustmentFactor * threadScalingFactor;
        double effectiveNPS = referenceNPS * totalScale;

        if (useNodesTime)
            effectiveNPS = double(npmsec) * 1000.0;

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
                double c_sd = 0.40 - 0.05 * std::tanh(effectiveInc / std::max(1.0, double(moveOverhead)));
                sdScale = (u * (1.0 + c_sd)) / (u + c_sd);
                M       = 2.0 + (M_pieces - 2.0) * u;
            }
        }

        // Remaining game duration across our physically anchored horizon M.
        // Dimensionally grounded total effective time without artificial overdraft inflation.
        double totalExpectedMs    = double(limits.time[us]) + (M - 1.0) * effectiveInc;
        double totalEffectiveTime = std::max(1.0, totalExpectedMs);
        double remainingGameMs    = totalEffectiveTime / double(scaleFactor);
        double remainingGameSec   = std::max(0.01, remainingGameMs / 1000.0);

        double effectiveGameSec = remainingGameSec * (effectiveNPS / referenceNPS);
        double tauRawActual     = std::log10(effectiveGameSec);

        // 2. Front-loading intensity tau via calibrated sigmoid (k = 1.55, x0 = 1.25) into 2nd-order polynomial:
        // P(s) = c1 * s + (1.0 - c1) * s^2 where s in [0, 1]
        // Provides steepened transition (dtau/dx ~ 1.00 - 1.35 across 5s-15s) with elevated bullet SD floor (tau ~ 1.34)
        // and generous headroom for VVLTC (tau ~ 4.25) and Classical TCs without premature saturation.
        constexpr double deltaTau = 3.80;
        constexpr double k        = 1.55;
        constexpr double x0       = 1.25;
        constexpr double c1       = 0.60;

        double s = 0.0;
        if (tauRawActual > -2.0)
            s = 1.0 / (1.0 + std::exp(-k * (tauRawActual - x0)));

        double p_s = c1 * s + (1.0 - c1) * (s * s);
        double tau = 1.0 + deltaTau * p_s;

        // 3. Time Bank & Nominal Draw with Discrete Renewal Horizon
        // Time bank deducts safety reserve and the current move's incoming increment
        TimePoint safetyReserve = moveOverhead * 10;
        TimePoint timeBank =
          std::max(TimePoint(0), limits.time[us] - safetyReserve - TimePoint(std::max(0.0, effectiveInc)));

        // Multiplicative bank factor governed by dynamic horizon M
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

        // 4. Dual early ply decay factors
        // - decayBank: steeper recovery for bank draw (fast recovery for SD / low inc)
        // - decayInc: flatter profile for increment discount (lingering longer to bank high inc)
        constexpr double c_bank = 0.065;
        constexpr double p_bank = 1.20;
        double decayBank = std::exp(-c_bank * std::pow(double(ply), p_bank));

        constexpr double c_inc  = 0.016;
        constexpr double p_inc  = 1.00;
        double decayInc  = std::exp(-c_inc * std::pow(double(ply), p_inc));

        // 5. Branchless component discounting (zero kinks)
        double r_bank = 0.52 + 0.05 * p_s;
        constexpr double r_inc  = 0.20;

        double bankComponent = bankDraw * (1.0 - r_bank * decayBank);
        double incComponent  = effectiveInc * (1.0 - r_inc * decayInc);
        double moveBudget    = std::max(0.0, (bankComponent + incComponent) * sdScale);

        TimePoint nominalOptimum = std::max(TimePoint(1), TimePoint(moveBudget));

        // 6. Dynamic maxScale ceiling grounded to current remaining clock
        double logTimeInSec      = std::log10(std::max(0.001, double(limits.time[us]) / (1000.0 * double(scaleFactor))));
        double maxConstant       = std::max(3.3744 + 3.0608 * logTimeInSec, 3.1441);
        double maxScale          = std::min(6.873, maxConstant + ply / 12.352);
        TimePoint nominalMaximum = std::max(nominalOptimum, TimePoint(nominalOptimum * maxScale));

        // 7. Safety buffer zone: scale down smoothly as timeBank empties
        // Uses timeBank with threshold = 1.50 * nominalMaximum and cubic curvature (1 - (1 - u)^3)
        // guaranteeing zero slope (no kinks) at the threshold while preserving endgame search depth.
        optimumTime = nominalOptimum;
        maximumTime = nominalMaximum;
        double safetyThreshold = 1.50 * double(nominalMaximum);
        if (safetyThreshold > 0.0 && double(timeBank) < safetyThreshold)
        {
            double u     = std::clamp(double(timeBank) / safetyThreshold, 0.0, 1.0);
            double v     = 1.0 - u;
            double scale = 1.0 - v * v * v;
            optimumTime  = std::max(TimePoint(1), TimePoint(double(nominalOptimum) * scale));
            maximumTime  = std::max(optimumTime,  TimePoint(double(nominalMaximum) * scale));
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
