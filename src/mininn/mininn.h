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
    bool is_mp_enabled() const { return is_loaded() && use_mp.load(std::memory_order_relaxed); }
    bool is_lmr_enabled() const { return is_loaded() && use_lmr.load(std::memory_order_relaxed); }
    void set_use_mp(bool val) { use_mp.store(val, std::memory_order_relaxed); }
    void set_use_lmr(bool val) { use_lmr.store(val, std::memory_order_relaxed); }

    // Feature extraction helpers (Shared between inference and telemetry serialization)
    static void extract_node_features(
        const Position& pos,
        const Search::Stack* ss,
        bool improving,
        bool cutNode,
        bool pvNode,
        int8_t out_u[MiniNN::NODE_IN_DIM]
    );

    static void extract_quiet_features(
        const Position& pos,
        Move m,
        const Search::Stack* ss,
        const ButterflyHistory* mainHistory,
        const LowPlyHistory* lowPlyHistory,
        const PieceToHistory** continuationHistory,
        const SharedHistories* sharedHistory,
        const Bitboard* threatByLesser,
        int ply,
        int8_t out_x[MiniNN::QUIET_IN_DIM]
    );

    static void extract_lmr_features(
        Move m,
        Piece movedPiece,
        bool is_capture,
        Piece capturedPiece,
        bool givesCheck,
        int moveCount,
        const Search::Stack* ss,
        int8_t out_x[MiniNN::LMR_IN_DIM]
    );

    // Evaluated ONCE per search node: populates ss->miniNN_w_quiet, ss->miniNN_w_lmr, and temperatures
    void evaluate_node(const Position& pos, Search::Stack* ss, bool improving, bool cutNode, bool pvNode) const;

private:
    std::atomic<bool> loaded{false};
    std::atomic<bool> use_mp{true};
    std::atomic<bool> use_lmr{true};

    // Node Network: fc0 (16 -> 32), fc1 (32 -> 32), fc2 (32 -> 18)
    alignas(32) int32_t node_b0[MiniNN::NODE_H_DIM];
    alignas(32) int8_t  node_w0[MiniNN::NODE_H_DIM][MiniNN::NODE_IN_DIM];
    alignas(32) int32_t node_b1[MiniNN::NODE_H_DIM];
    alignas(32) int8_t  node_w1[MiniNN::NODE_H_DIM][MiniNN::NODE_H_DIM];
    alignas(32) int32_t node_b2[MiniNN::NODE_OUT_DIM];
    alignas(32) int8_t  node_w2[MiniNN::NODE_OUT_DIM][MiniNN::NODE_H_DIM];
};

extern MiniNNModel globalMiniNN;

} // namespace Stockfish

#endif // #ifndef MININN_H_INCLUDED
