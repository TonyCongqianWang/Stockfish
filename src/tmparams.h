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

#ifndef TMPARAMS_H_INCLUDED
#define TMPARAMS_H_INCLUDED

namespace Stockfish::TimeParams {

// Fixed-point coefficients: divide by 1000 unless the use states otherwise.
inline int TMHorizonBase       = 10000;
inline int TMHorizonPiece      = 2000;
inline int TMBankRate          = 1800;
inline int TMIncRate           = 1000;
inline int TMOpeningFloor      = 550;
inline int TMOpeningPlies      = 18000;
inline int TMReserveMoves      = 2000;
inline int TMMaxSD             = 5000;
inline int TMMaxInc            = 6873;
inline int TMElasticitySD      = 1000;
inline int TMElasticityInc     = 1000;
inline int TMBehind            = 300;
inline int TMClockCap          = 810;
inline int TMCyclicBase        = 880;
inline int TMCyclicPly         = 116400;
inline int TMCyclicMaxBase     = 1300;
inline int TMCyclicMaxMove     = 110;
inline int TMCyclicBehind      = 900;
inline int TMEvalBase          = 11480;
inline int TMEvalPrevious      = 2300;
inline int TMEvalIteration     = 1100;
inline int TMEvalMin           = 576;
inline int TMEvalMax           = 1728;
inline int TMStableStart       = 4960;
inline int TMStableSpan        = 13830;
inline int TMStableLow         = 639;
inline int TMStableRise        = 1073;
inline int TMStableMin         = 629;
inline int TMStableMax         = 1544;
inline int TMReductionOffset   = 1468;
inline int TMReductionDivisor  = 2284;
inline int TMChangeDecay       = 500;
inline int TMChangeBase        = 1077;
inline int TMChangeWeight      = 2229;
inline int TMEffortStart       = 75800;
inline int TMEffortSpan        = 28710;
inline int TMEffortBase        = 969;
inline int TMEffortDrop        = 255;
inline int TMEffortMin         = 693;
inline int TMEffortMax         = 838;
inline int TMDepthFraction     = 500;
inline int TMPreviousReduction = 850;
inline int TMPonderBonus       = 250;

}  // namespace Stockfish::TimeParams

#endif
