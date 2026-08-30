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

#ifndef MININN_TYPES_H_INCLUDED
#define MININN_TYPES_H_INCLUDED

#include <cstdint>

namespace Stockfish {
namespace MiniNN {

static constexpr uint32_t MAGIC = 0x4D494E49; // 'MINI'
static constexpr uint32_t VERSION = 5;        // Version 5: 8 MovePicker Terms (Scale 256) + 8 LMR Terms (Scale 64)

static constexpr int WEIGHT_SCALE = 64;

// 1. Node Network
static constexpr int NODE_IN_DIM = 16;
static constexpr int NODE_H_DIM = 32;
static constexpr int QUIET_TERMS = 8;         // 8 dynamic weights for handcrafted quiet move terms (Scale 256)
static constexpr int LMR_TERMS = 8;           // 8 dynamic residual coefficient tweaks for LMR (Scale 64)
static constexpr int NODE_OUT_DIM = QUIET_TERMS + LMR_TERMS; // 8 + 8 = 16

// 2. Quiet Move Features & LMR Features (8 Handcrafted Terms Each)
static constexpr int QUIET_IN_DIM = 8;
static constexpr int LMR_IN_DIM = 8;

} // namespace MiniNN
} // namespace Stockfish

#endif // #ifndef MININN_TYPES_H_INCLUDED
