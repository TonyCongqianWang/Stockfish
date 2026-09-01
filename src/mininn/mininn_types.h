/*
  Mini-NN Type Definitions and Architecture Dimensions
*/

#ifndef MININN_TYPES_H_INCLUDED
#define MININN_TYPES_H_INCLUDED

#include <cstdint>

namespace Stockfish {
namespace MiniNN {

static constexpr uint32_t MAGIC = 0x4D494E49; // 'MINI'
static constexpr uint32_t VERSION = 4;        // Version 4: Dynamic Quiet Terms (Scale 256) + LMR

static constexpr int WEIGHT_SCALE = 64;

// 1. Node Network
static constexpr int NODE_IN_DIM = 16;
static constexpr int NODE_H_DIM = 32;
static constexpr int QUIET_TERMS = 10;        // 10 dynamic weights for handcrafted quiet move terms
static constexpr int NODE_LATENTS = 8;        // 8 position latents for evaluate_lmr
static constexpr int NODE_OUT_DIM = QUIET_TERMS + NODE_LATENTS + 2; // 10 + 8 + 1 (tau_mp) + 1 (tau_lmr) = 20

// 2. Quiet Move Features (10 Handcrafted Terms)
static constexpr int QUIET_IN_DIM = 10;

// 3. LMR Network (Search Depth Reduction with Node Latents)
static constexpr int LMR_IN_DIM = 16; // 8 candidate move features + 8 node latents
static constexpr int LMR_H_DIM = 16;

} // namespace MiniNN
} // namespace Stockfish

#endif // #ifndef MININN_TYPES_H_INCLUDED
