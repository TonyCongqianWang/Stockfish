/*
  Mini-NN: Neural Move Ordering (score_quiet, score_capture) & Search Reductions (evaluate_lmr)
  Symmetric Quantized Inference in [-127, 127]
*/

#ifndef MININN_H_INCLUDED
#define MININN_H_INCLUDED

#include <cstdint>
#include <string>
#include <atomic>

#include "../types.h"
#include "../bitboard.h"
#include "../history.h"
#include "mininn_types.h"

namespace Stockfish {

class Position;
namespace Search { struct Stack; }

class MiniNNModel {
public:
    MiniNNModel();
    ~MiniNNModel() = default;

    bool load(const std::string& filepath);
    bool is_loaded() const { return loaded.load(std::memory_order_acquire); }

    // Evaluated ONCE per search node: populates ss->miniNN_w_quiet, ss->miniNN_z_latents, and temperatures
    void evaluate_node(const Position& pos, Search::Stack* ss, bool improving, bool cutNode, bool pvNode) const;

    // Evaluated for quiet moves in MovePicker (replaces handcrafted history sum)
    int score_quiet(
        const Position& pos,
        Move m,
        const Search::Stack* ss,
        const ButterflyHistory* mainHistory,
        const LowPlyHistory* lowPlyHistory,
        const PieceToHistory** continuationHistory,
        const SharedHistories* sharedHistory,
        const Bitboard* threatByLesser,
        int ply
    ) const;

    // Evaluated for captures in MovePicker (replaces handcrafted capture score)
    int score_capture(
        const Position& pos,
        Move m,
        const Search::Stack* ss,
        const CapturePieceToHistory* captureHistory
    ) const;

    // Evaluated in search.cpp Step 18: outputs LMR reduction delta in 1024 fixed-point scale
    int evaluate_lmr(
        const Position& pos,
        Move m,
        int moveCount,
        const Search::Stack* ss
    ) const;

private:
    std::atomic<bool> loaded{false};

    // 1. Node Network: fc0 (16 -> 32), fc1 (32 -> 32), fc2 (32 -> 26)
    alignas(32) int32_t node_b0[MiniNN::NODE_H_DIM];
    alignas(32) int8_t  node_w0[MiniNN::NODE_H_DIM][MiniNN::NODE_IN_DIM];
    alignas(32) int32_t node_b1[MiniNN::NODE_H_DIM];
    alignas(32) int8_t  node_w1[MiniNN::NODE_H_DIM][MiniNN::NODE_H_DIM];
    alignas(32) int32_t node_b2[MiniNN::NODE_OUT_DIM];
    alignas(32) int8_t  node_w2[MiniNN::NODE_OUT_DIM][MiniNN::NODE_H_DIM];

    // 2. Quiet Move Network: fc0 (12 -> 16)
    alignas(32) int32_t quiet_b0[MiniNN::QUIET_H_DIM];
    alignas(32) int8_t  quiet_w0[MiniNN::QUIET_H_DIM][MiniNN::QUIET_IN_DIM];

    // 3. Capture Move Network: fc0 (12 -> 16), fc1 (16 -> 1)
    alignas(32) int32_t cap_b0[MiniNN::CAPTURE_H_DIM];
    alignas(32) int8_t  cap_w0[MiniNN::CAPTURE_H_DIM][MiniNN::CAPTURE_IN_DIM];
    alignas(32) int32_t cap_b1[1];
    alignas(32) int8_t  cap_w1[1][MiniNN::CAPTURE_H_DIM];

    // 4. LMR Network: fc0 (16 -> 16), fc1 (16 -> 1)
    alignas(32) int32_t lmr_b0[MiniNN::LMR_H_DIM];
    alignas(32) int8_t  lmr_w0[MiniNN::LMR_H_DIM][MiniNN::LMR_IN_DIM];
    alignas(32) int32_t lmr_b1[1];
    alignas(32) int8_t  lmr_w1[1][MiniNN::LMR_H_DIM];
};

extern MiniNNModel globalMiniNN;

} // namespace Stockfish

#endif // #ifndef MININN_H_INCLUDED
