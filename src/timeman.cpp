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
#include "tmparams.h"
#include "types.h"
#include "ucioption.h"

namespace Stockfish {

namespace TimeParams {
TUNE(SetRange(2000, 30000), TMHorizonBase);
TUNE(SetRange(250, 4000), TMHorizonPiece);
TUNE(SetRange(300, 4000), TMBankRate);
TUNE(SetRange(300, 1500), TMIncRate);
TUNE(SetRange(200, 1000), TMOpeningFloor);
TUNE(SetRange(2000, 60000), TMOpeningPlies);
TUNE(SetRange(0, 15000), TMReserveMoves);
TUNE(SetRange(1000, 10000), TMMaxSD);
TUNE(SetRange(1000, 12000), TMMaxInc);
TUNE(SetRange(100, 2000), TMElasticitySD);
TUNE(SetRange(100, 2000), TMElasticityInc);
TUNE(SetRange(0, 1200), TMBehind);
TUNE(SetRange(300, 950), TMClockCap);
TUNE(SetRange(200, 1500), TMCyclicBase);
TUNE(SetRange(20000, 300000), TMCyclicPly);
TUNE(SetRange(1000, 3000), TMCyclicMaxBase);
TUNE(SetRange(10, 250), TMCyclicMaxMove);
TUNE(SetRange(0, 1200), TMCyclicBehind);
TUNE(SetRange(1000, 50000), TMEvalBase);
TUNE(SetRange(100, 6000), TMEvalPrevious);
TUNE(SetRange(100, 4000), TMEvalIteration);
TUNE(SetRange(100, 1000), TMEvalMin);
TUNE(SetRange(1000, 4000), TMEvalMax);
TUNE(SetRange(1000, 10000), TMStableStart);
TUNE(SetRange(1000, 30000), TMStableSpan);
TUNE(SetRange(100, 1200), TMStableLow);
TUNE(SetRange(100, 2500), TMStableRise);
TUNE(SetRange(100, 1000), TMStableMin);
TUNE(SetRange(1000, 3000), TMStableMax);
TUNE(SetRange(100, 3000), TMReductionOffset);
TUNE(SetRange(500, 5000), TMReductionDivisor);
TUNE(SetRange(0, 950), TMChangeDecay);
TUNE(SetRange(300, 2000), TMChangeBase);
TUNE(SetRange(100, 6000), TMChangeWeight);
TUNE(SetRange(10000, 95000), TMEffortStart);
TUNE(SetRange(1000, 60000), TMEffortSpan);
TUNE(SetRange(300, 1500), TMEffortBase);
TUNE(SetRange(0, 700), TMEffortDrop);
TUNE(SetRange(100, 1000), TMEffortMin);
TUNE(SetRange(500, 1500), TMEffortMax);
TUNE(SetRange(100, 900), TMDepthFraction);
TUNE(SetRange(100, 2000), TMPreviousReduction);
TUNE(SetRange(0, 1000), TMPonderBonus);
}  // namespace TimeParams

using namespace TimeParams;

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
                          const OptionsMap&   options) {
    Color us         = pos.side_to_move();
    int   ply        = pos.game_ply();
    dynamicExponent  = 1.0;
    TimePoint npmsec = TimePoint(options["nodestime"]);

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

    double budget, maxScale;
    if (limits.movestogo == 0)
    {
        // Allocate the current bank over a material-dependent horizon. Increment is
        // renewable income; overhead is paid on every move, including sudden death.
        double horizon =
          std::max(1.0, (TMHorizonBase + TMHorizonPiece * pos.count<ALL_PIECES>()) / 1000.0);
        double income = double(limits.inc[us]) - moveOverhead;
        double bank =
          std::max(0.0, double(limits.time[us]) - moveOverhead * (TMReserveMoves / 1000.0));
        double bankDraw = (TMBankRate / 1000.0) * bank / horizon;

        // Only bank spending responds to the opponent clock. Income remains usable.
        if (!useNodesTime)
        {
            double advantage = (double(limits.time[us]) - limits.time[~us])
                             / (1.0 + double(limits.time[us]) + limits.time[~us]);
            bankDraw *= std::max(0.0, 1.0 + TMBehind / 1000.0 * std::min(advantage, 0.0));
        }

        double opening =
          TMOpeningFloor / 1000.0
          + (1.0 - TMOpeningFloor / 1000.0) * std::tanh(ply / (TMOpeningPlies / 1000.0));
        budget = std::max(0.0, bankDraw + TMIncRate / 1000.0 * income) * opening;

        // Renewable share is dimensionless and independent of the initial clock
        // and machine speed. Share the signal between headroom and responsiveness.
        double renewable = horizon * std::max(0.0, income);
        double share     = renewable / std::max(1.0, bank + renewable);
        maxScale         = (TMMaxSD + (TMMaxInc - TMMaxSD) * share) / 1000.0;
        dynamicExponent  = (TMElasticitySD + (TMElasticityInc - TMElasticitySD) * share) / 1000.0;
    }
    else
    {
        // Keep master's cyclic allocation as the starting point.
        int    mtg      = std::min(limits.movestogo, 50);
        double timeLeft = std::max(1.0, double(limits.time[us]) + double(limits.inc[us]) * (mtg - 1)
                                          - double(moveOverhead) * (2 + mtg));
        double scale    = std::min((TMCyclicBase / 1000.0 + ply / (TMCyclicPly / 1000.0)) / mtg,
                                   TMCyclicBase / 1000.0 * limits.time[us] / timeLeft);
        if (!useNodesTime && limits.movestogo != 1)
        {
            double advantage = (double(limits.time[us]) - limits.time[~us])
                             / (1.0 + double(limits.time[us]) + limits.time[~us]);
            scale *= std::max(0.0, 1.0 + TMCyclicBehind / 1000.0 * std::min(advantage, 0.0));
        }
        budget   = scale * timeLeft;
        maxScale = (TMCyclicMaxBase + TMCyclicMaxMove * mtg) / 1000.0;
    }

    // Safety is independent of responsiveness: no extrapolation past the clock cap.
    double cap  = std::max(1.0, TMClockCap / 1000.0 * limits.time[us] - moveOverhead);
    optimumTime = TimePoint(std::clamp(budget, 1.0, cap));
    maximumTime = TimePoint(std::clamp(maxScale * optimumTime, double(optimumTime), cap));

    if (options["Ponder"])
        optimumTime += TimePoint(optimumTime * (TMPonderBonus / 1000.0));
}

}  // namespace Stockfish
