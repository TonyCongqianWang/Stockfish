/*
  Mini-NN Type Definitions and Architecture Dimensions
*/

#ifndef MININN_TYPES_H_INCLUDED
#define MININN_TYPES_H_INCLUDED

#include <cstdint>

namespace Stockfish {
namespace MiniNN {

static constexpr uint32_t MAGIC = 0x4D494E49; // 'MINI'
static constexpr uint32_t VERSION = 3;        // Version 3: Full Architecture (score_quiet, score_capture, evaluate_lmr)

static constexpr int WEIGHT_SCALE = 64;

// 1. Node Network
static constexpr int NODE_IN_DIM = 16;
static constexpr int NODE_H_DIM = 32;
static constexpr int QUIET_META_WEIGHTS = 16; // 16 dynamic weights for score_quiet
static constexpr int NODE_LATENTS = 8;        // 8 latents for score_capture & evaluate_lmr
static constexpr int NODE_OUT_DIM = QUIET_META_WEIGHTS + NODE_LATENTS + 2; // 16 + 8 + 1 (tau_mp) + 1 (tau_lmr) = 26

// 2. Quiet Move Network (Meta-Learned History Combiner)
static constexpr int QUIET_IN_DIM = 12;
static constexpr int QUIET_H_DIM = 16;

// 3. Capture Move Network (Tactical Combiner with Node Latents)
static constexpr int CAPTURE_IN_DIM = 12; // 4 raw capture signals + 8 node latents
static constexpr int CAPTURE_H_DIM = 16;

// 4. LMR Network (Search Depth Reduction with Node Latents)
static constexpr int LMR_IN_DIM = 16; // 8 candidate move features + 8 node latents
static constexpr int LMR_H_DIM = 16;

} // namespace MiniNN
} // namespace Stockfish

#endif // #ifndef MININN_TYPES_H_INCLUDED
