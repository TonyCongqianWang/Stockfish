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
                          double&             initialGameSec,
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
    const i64       scaleFactor = useNodesTime ? npmsec : 1;
    const TimePoint scaledTime  = std::max(TimePoint(1), limits.time[us] / scaleFactor);

    // Sudden death (no moves to go specified)
    if (limits.movestogo == 0)
    {
        // Net per-move increment cash flow after accounting for communication/execution overhead.
        // Allowed to be negative in sudden death / low increment formats, naturally deducting overhead.
        double effectiveInc = double(limits.inc[us] - moveOverhead);

        // Characteristic total game duration scale: tau = log10(effectiveGameSec)
        // Scaled by effective compute effort across threads and hardware speed
        if (initialGameSec <= 0.0)
        {
            double effectiveIncMs = effectiveInc / double(scaleFactor);
            double sublinear =
              std::copysign(std::pow(std::abs(effectiveIncMs), 0.10), effectiveIncMs);
            double linearGameMs =
              (double(limits.time[us]) + 49.0 * effectiveInc - 6.0 * double(moveOverhead)) / double(scaleFactor);
            initialGameSec = std::max(0.1, (linearGameMs + 50.0 * sublinear) / 1000.0);
        }

        int          threadsCount = std::max(1, int(options["Threads"]));
        const double referenceNPS = 628'000.0;
        double       effectiveNPS = referenceNPS * std::pow(threadsCount, 0.85);

        if (useNodesTime)
            effectiveNPS = double(npmsec) * 1000.0;
        else if (totalGameTimeMs >= 100)
            effectiveNPS = (double(totalGameNodes) * 1000.0) / double(totalGameTimeMs);

        double effectiveGameSec = initialGameSec * (effectiveNPS / referenceNPS);
        double tauRaw           = std::log10(std::max(1.0, effectiveGameSec));
        double tau              = 1.85 + 0.71 * std::pow(tauRaw, 1.20);

        // 1. Physically anchored remaining moves horizon (M = 4 + 2 * pieces)
        double M = std::max(8.0, 4.0 + 2.0 * pos.count<ALL_PIECES>());

        // 2. Time Bank & Nominal Draw
        TimePoint safetyReserve   = moveOverhead * 4;
        TimePoint timeBank        = std::max(TimePoint(0), limits.time[us] - safetyReserve);
        double    nominalBankDraw = double(timeBank) / M;

        // 3. Multiplicative bank factor with concave material fraction and king baseline
        constexpr double KingValue   = QueenValue * 1.25;
        constexpr double MaxMaterial = 2.0 * (QueenValue + 2 * RookValue + 2 * BishopValue + 2 * KnightValue + 8 * PawnValue + KingValue);

        double totalMat = double(pos.non_pawn_material()) + pos.count<PAWN>() * double(PawnValue) + 2.0 * KingValue;
        double matFrac  = std::clamp(totalMat / MaxMaterial, 0.0, 1.0);
        double tbFactor = std::max(0.50, tau * std::pow(matFrac, 0.55));
        double bankDraw = nominalBankDraw * tbFactor;

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

        // 4. Base move budget combining time bank draw and net increment cash flow
        double baseMoveBudget = bankDraw + effectiveInc;

        // 5. Early ply discount applied to base budget with asymptotic saturation and book half-life
        double p_ply = 1.20 + 0.42 * tauRaw;
        double A_ply = 1.0 - 1.10 / (1.50 + 0.60 * tau);
        double w_ply = 1.0 - A_ply * std::pow(12.0 / (12.0 + ply), p_ply);
        optimumTime  = std::max(TimePoint(1), TimePoint(baseMoveBudget * w_ply));

        // 6. Gentle discount (up to 20%) when time left is smaller than 2.5x optimum time to avoid paycheck-to-paycheck trap
        if (double(limits.time[us]) < 2.5 * double(optimumTime) && optimumTime > 0)
        {
            double discount = 0.20 * (1.0 - double(limits.time[us]) / (2.5 * double(optimumTime)));
            optimumTime     = std::max(TimePoint(1), TimePoint(double(optimumTime) * (1.0 - discount)));
        }

        // 7. Dynamic maxScale ceiling and hard safety caps
        double logTimeInSec = std::log10(scaledTime / 1000.0);
        double maxConstant  = std::max(3.3744 + 3.0608 * logTimeInSec, 3.1441);
        double maxScale     = std::min(6.873, maxConstant + ply / 12.352);

        TimePoint maxClockCap = std::max(TimePoint(1), TimePoint(0.8097 * limits.time[us] - moveOverhead));
        optimumTime           = std::min(optimumTime, maxClockCap);
        maximumTime           = std::max(optimumTime, std::min(maxClockCap, TimePoint(optimumTime * maxScale)));
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
