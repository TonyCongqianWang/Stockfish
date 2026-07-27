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

//Definition of input features QK4 of NNUE evaluation function

#include "qk4.h"

namespace Stockfish::Eval::NNUE::Features {

void QK4::append_changed_indices(
  Color perspective, Square ksq, const DiffType& diff, bool opponent_has_queen, IndexList& removed, IndexList& added) {
    (void) perspective;
    (void) ksq;
    (void) diff;
    (void) opponent_has_queen;
    (void) removed;
    (void) added;
}

bool QK4::requires_refresh(const DiffType& diff, Color perspective) {
    (void) diff;
    (void) perspective;
    return false;
}

}  // namespace Stockfish::Eval::NNUE::Features
