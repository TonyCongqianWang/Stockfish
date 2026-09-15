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
                          double&             originalTimeAdjust) {
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
        // Characteristic total game duration scale: tau = log10(totalGameSec)
        // Based on empirical average game length of 65 moves
        if (originalTimeAdjust < 0)
        {
            double totalGameSec =
              (scaledTime + (limits.inc[us] / scaleFactor) * 65) / 1000.0;
            originalTimeAdjust = std::log10(std::max(1.0, totalGameSec));
        }
        double tau = originalTimeAdjust;

        // 1. Physically anchored remaining moves horizon (M = 4 + 2 * pieces)
        double M = std::max(8.0, 4.0 + 2.0 * pos.count<ALL_PIECES>());

        // 2. Time Bank & Nominal Draw
        TimePoint safetyReserve   = moveOverhead * 2;
        TimePoint timeBank        = std::max(TimePoint(0), limits.time[us] - limits.inc[us] - safetyReserve);
        double    nominalBankDraw = double(timeBank) / M;

        // 3. Bank draw with flat baseline (1.0) and tau-dependent complexity overdraft
        double totalMat     = double(pos.non_pawn_material()) + pos.count<PAWN>() * 208.0;
        double matFrac      = std::clamp((totalMat - 1000.0) / (19932.0 - 1000.0), 0.0, 1.0);
        double maxOverdraft = std::clamp(0.60 + 0.60 * (tau - 1.22), 0.20, 1.20);
        double f_bank       = 1.0 + maxOverdraft * matFrac;
        double bankDraw     = nominalBankDraw * f_bank;

        // 4. Base move budget combining overdrafted bank draw and increment
        double effectiveInc   = std::min(double(limits.inc[us]), double(limits.time[us]));
        double baseMoveBudget = bankDraw + effectiveInc;

        // 5. Early ply discount applied to base budget (enables banking increment early)
        double w_ply    = 1.0 - 0.25 * (24.0 / (24.0 + ply));
        double ohDeduct = std::min(double(moveOverhead), effectiveInc);
        optimumTime     = std::max(TimePoint(1), TimePoint(baseMoveBudget * w_ply - ohDeduct));

        // 6. Decrease time usage if behind in time
        if (!useNodesTime)
        {
            double timeAdvantage =
              (limits.time[us] - limits.time[~us]) / (1.0 + limits.time[us] + limits.time[~us]);
            optimumTime =
              std::max(TimePoint(1), TimePoint(optimumTime * (1.0 + 0.9 * std::min(timeAdvantage, 0.0))));
        }

        // 7. Dynamic maxScale ceiling and hard safety caps
        double logTimeInSec = std::log10(scaledTime / 1000.0);
        double maxConstant  = std::max(3.3744 + 3.0608 * logTimeInSec, 3.1441);
        double maxScale     = std::min(6.873, maxConstant + ply / 12.352);

        TimePoint maxClockCap = TimePoint(0.8097 * limits.time[us] - moveOverhead);
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
