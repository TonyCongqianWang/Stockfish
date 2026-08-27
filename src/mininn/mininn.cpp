/*
  Mini-NN Engine Implementation:
  score_quiet (Meta-Learned History Combiner)
  score_capture (Tactical Combiner with Position Latents)
  evaluate_lmr (Search Depth Reductions)
*/

#include "mininn.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstring>

#include "../position.h"
#include "../search.h"

namespace Stockfish {

MiniNNModel globalMiniNN;

MiniNNModel::MiniNNModel() {
    std::memset(node_b0, 0, sizeof(node_b0));
    std::memset(node_w0, 0, sizeof(node_w0));
    std::memset(node_b1, 0, sizeof(node_b1));
    std::memset(node_w1, 0, sizeof(node_w1));
    std::memset(node_b2, 0, sizeof(node_b2));
    std::memset(node_w2, 0, sizeof(node_w2));

    std::memset(quiet_b0, 0, sizeof(quiet_b0));
    std::memset(quiet_w0, 0, sizeof(quiet_w0));

    std::memset(cap_b0, 0, sizeof(cap_b0));
    std::memset(cap_w0, 0, sizeof(cap_w0));
    std::memset(cap_b1, 0, sizeof(cap_b1));
    std::memset(cap_w1, 0, sizeof(cap_w1));

    std::memset(lmr_b0, 0, sizeof(lmr_b0));
    std::memset(lmr_w0, 0, sizeof(lmr_w0));
    std::memset(lmr_b1, 0, sizeof(lmr_b1));
    std::memset(lmr_w1, 0, sizeof(lmr_w1));
}

bool MiniNNModel::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
        return false;

    uint32_t header[8];
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file || header[0] != MiniNN::MAGIC || header[1] != MiniNN::VERSION)
        return false;

    // 1. Node Network: fc0 (16 -> 32), fc1 (32 -> 32), fc2 (32 -> 26)
    file.read(reinterpret_cast<char*>(node_b0), sizeof(node_b0));
    file.read(reinterpret_cast<char*>(node_w0), sizeof(node_w0));

    file.read(reinterpret_cast<char*>(node_b1), sizeof(node_b1));
    file.read(reinterpret_cast<char*>(node_w1), sizeof(node_w1));

    file.read(reinterpret_cast<char*>(node_b2), sizeof(node_b2));
    file.read(reinterpret_cast<char*>(node_w2), sizeof(node_w2));

    // 2. Quiet Network: fc0 (12 -> 16)
    file.read(reinterpret_cast<char*>(quiet_b0), sizeof(quiet_b0));
    file.read(reinterpret_cast<char*>(quiet_w0), sizeof(quiet_w0));

    // 3. Capture Network: fc0 (12 -> 16), fc1 (16 -> 1)
    file.read(reinterpret_cast<char*>(cap_b0), sizeof(cap_b0));
    file.read(reinterpret_cast<char*>(cap_w0), sizeof(cap_w0));

    file.read(reinterpret_cast<char*>(cap_b1), sizeof(cap_b1));
    file.read(reinterpret_cast<char*>(cap_w1), sizeof(cap_w1));

    // 4. LMR Network: fc0 (16 -> 16), fc1 (16 -> 1)
    file.read(reinterpret_cast<char*>(lmr_b0), sizeof(lmr_b0));
    file.read(reinterpret_cast<char*>(lmr_w0), sizeof(lmr_w0));

    file.read(reinterpret_cast<char*>(lmr_b1), sizeof(lmr_b1));
    file.read(reinterpret_cast<char*>(lmr_w1), sizeof(lmr_w1));

    if (!file)
        return false;

    loaded.store(true, std::memory_order_release);
    return true;
}

void MiniNNModel::evaluate_node(
    const Position& pos,
    Search::Stack* ss,
    bool improving,
    bool cutNode,
    bool pvNode
) const {
    if (!loaded.load(std::memory_order_relaxed) || !ss)
        return;

    Color us = pos.side_to_move();
    Color them = ~us;

    int npm_us = pos.non_pawn_material(us);
    int npm_them = pos.non_pawn_material(them);
    int pawns_us = pos.count<PAWN>(us);
    int pawns_them = pos.count<PAWN>(them);

    int num_pinned = popcount(pos.blockers_for_king(us));

    // 16 node inputs in [-127, 127]
    int8_t u[MiniNN::NODE_IN_DIM];
    u[0]  = int8_t(std::clamp((npm_us - 2000) * 64 / 1000, -127, 127));
    u[1]  = int8_t(std::clamp((npm_them - 2000) * 64 / 1000, -127, 127));
    u[2]  = int8_t(std::clamp((pawns_us - 4) * 64 / 2, -127, 127));
    u[3]  = int8_t(std::clamp((pawns_them - 4) * 64 / 2, -127, 127));
    u[4]  = cutNode ? 64 : -64;
    u[5]  = int8_t(std::clamp(num_pinned * 64 / 4, -127, 127));
    u[6]  = pvNode ? 64 : -64;
    u[7]  = ss->ttPv ? 64 : -64;
    u[8]  = int8_t(std::clamp((int(ss->ply) - 24) * 64 / 16, -127, 127));
    u[9]  = improving ? 64 : -64;
    u[10] = ss->ttHit ? 64 : -64;
    u[11] = ss->ttPv ? 64 : -64;
    u[12] = int8_t(std::clamp(int(ss->staticEval) * 64 / 500, -127, 127));
    u[13] = int8_t(std::clamp(int(ss->statScore) * 64 / 2000, -127, 127));
    u[14] = int8_t(std::clamp((ss->cutoffCnt - 1) * 64 / 2, -127, 127));
    u[15] = (npm_us + npm_them < 3000) ? 64 : -64;

    // Layer 0 (16 -> 32)
    alignas(32) int32_t h0[MiniNN::NODE_H_DIM];
    for (int j = 0; j < MiniNN::NODE_H_DIM; ++j)
    {
        int32_t sum = node_b0[j];
        for (int i = 0; i < MiniNN::NODE_IN_DIM; ++i)
            sum += node_w0[j][i] * u[i];
        h0[j] = std::clamp((sum + 32) >> 6, 0, 127);
    }

    // Layer 1 (32 -> 32)
    alignas(32) int32_t h1[MiniNN::NODE_H_DIM];
    for (int j = 0; j < MiniNN::NODE_H_DIM; ++j)
    {
        int32_t sum = node_b1[j];
        for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
            sum += node_w1[j][i] * h0[i];
        h1[j] = std::clamp((sum + 32) >> 6, 0, 127);
    }

    // Layer 2 (32 -> 26)
    // 0..15: w_quiet meta-weights for score_quiet (scale 127)
    for (int k = 0; k < MiniNN::QUIET_META_WEIGHTS; ++k)
    {
        int32_t sum = node_b2[k];
        for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
            sum += node_w2[k][i] * h1[i];
        ss->miniNN_w_quiet[k] = int8_t(std::clamp((sum + 16) >> 5, -127, 127));
    }

    // 16..23: z_latents for score_capture and evaluate_lmr (scale 64)
    for (int k = 0; k < MiniNN::NODE_LATENTS; ++k)
    {
        int32_t sum = node_b2[MiniNN::QUIET_META_WEIGHTS + k];
        for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
            sum += node_w2[MiniNN::QUIET_META_WEIGHTS + k][i] * h1[i];
        ss->miniNN_z_latents[k] = int8_t(std::clamp((sum + 32) >> 6, -127, 127));
    }

    // 24: log_tau_mp
    int32_t sum_tau_mp = node_b2[24];
    for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
        sum_tau_mp += node_w2[24][i] * h1[i];
    int log_tau_mp = std::clamp(sum_tau_mp / 4096, -64, 64);
    ss->miniNN_inv_tau_mp = std::clamp(1024 - (log_tau_mp * 16), 256, 4096);

    // 25: log_tau_lmr
    int32_t sum_tau_lmr = node_b2[25];
    for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
        sum_tau_lmr += node_w2[25][i] * h1[i];
    int log_tau_lmr = std::clamp(sum_tau_lmr / 4096, -64, 64);
    ss->miniNN_inv_tau_lmr = std::clamp(1024 - (log_tau_lmr * 16), 256, 4096);
}

int MiniNNModel::score_quiet(
    const Position& pos,
    Move m,
    const Search::Stack* ss,
    const ButterflyHistory* mainHistory,
    const LowPlyHistory* lowPlyHistory,
    const PieceToHistory** continuationHistory,
    const SharedHistories* sharedHistory,
    const Bitboard* threatByLesser,
    int ply
) const {
    if (!loaded.load(std::memory_order_relaxed) || !ss)
        return 0;

    Color us = pos.side_to_move();
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = pos.moved_piece(m);
    PieceType pt = type_of(pc);

    // 12 raw signals normalized to symmetric int8 [-127, 127]
    int8_t x[MiniNN::QUIET_IN_DIM];
    x[0]  = int8_t(std::clamp(int((*mainHistory)[us][m.raw()]) * 64 / 16384, -127, 127));
    x[1]  = sharedHistory ? int8_t(std::clamp(int(sharedHistory->pawn_entry(pos)[pc][to]) * 64 / 16384, -127, 127)) : 0;
    x[2]  = continuationHistory && continuationHistory[0] ? int8_t(std::clamp(int((*continuationHistory[0])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    x[3]  = continuationHistory && continuationHistory[1] ? int8_t(std::clamp(int((*continuationHistory[1])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    x[4]  = continuationHistory && continuationHistory[2] ? int8_t(std::clamp(int((*continuationHistory[2])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    x[5]  = continuationHistory && continuationHistory[3] ? int8_t(std::clamp(int((*continuationHistory[3])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    x[6]  = continuationHistory && continuationHistory[5] ? int8_t(std::clamp(int((*continuationHistory[5])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    x[7]  = ((pos.check_squares(pt) & to) && pos.see_ge(m, -75)) ? 64 : -64;
    x[8]  = threatByLesser && (threatByLesser[pt] & from) ? 64 : -64;
    x[9]  = threatByLesser && (threatByLesser[pt] & to) ? 64 : -64;
    x[10] = int8_t(std::clamp((int(pt) - 2) * 64 / 2, -127, 127));
    x[11] = (ply < LOW_PLY_HISTORY_SIZE && lowPlyHistory) ? int8_t(std::clamp(int((*lowPlyHistory)[ply][m.raw()] / (1 + ply)) * 64 / 16384, -127, 127)) : 0;

    // Layer 0 (12 -> 16)
    alignas(32) int32_t h[MiniNN::QUIET_H_DIM];
    for (int j = 0; j < MiniNN::QUIET_H_DIM; ++j)
    {
        int32_t sum = quiet_b0[j];
        for (int i = 0; i < MiniNN::QUIET_IN_DIM; ++i)
            sum += quiet_w0[j][i] * x[i];
        h[j] = std::clamp((sum + 32) >> 6, 0, 127);
    }

    // Dynamic inner product with ss->miniNN_w_quiet[0..15]
    int32_t score_sum = 0;
    for (int k = 0; k < MiniNN::QUIET_H_DIM; ++k)
        score_sum += h[k] * ss->miniNN_w_quiet[k];

    // Scale to score units: (score_sum * 1200 + 4064) / (64 * 127) and clamp to [-1200, 1200]
    int score = (score_sum * 1200 + 4064) / (64 * 127);
    return std::clamp(score, -1200, 1200);
}

int MiniNNModel::score_capture(
    const Position& pos,
    Move m,
    const Search::Stack* ss,
    const CapturePieceToHistory* captureHistory
) const {
    if (!loaded.load(std::memory_order_relaxed) || !ss)
        return 0;

    Piece pc = pos.moved_piece(m);
    Piece capturedPiece = pos.piece_on(m.to_sq());

    // 4 raw capture signals + 8 position latents = 12 inputs (all at scale 64)
    int8_t x[MiniNN::CAPTURE_IN_DIM];
    x[0] = captureHistory ? int8_t(std::clamp(int((*captureHistory)[pc][m.to_sq()][type_of(capturedPiece)]) * 64 / 16384, -127, 127)) : 0;
    x[1] = int8_t(std::clamp(int(PieceValue[capturedPiece]) * 64 / 500, 0, 127));
    x[2] = int8_t(std::clamp((int(PieceValue[capturedPiece]) - int(PieceValue[pc])) * 64 / 500, -127, 127));
    x[3] = pos.gives_check(m) ? 64 : -64;

    // 8 Position Latents (scale 64)
    for (int k = 0; k < MiniNN::NODE_LATENTS; ++k)
        x[4 + k] = ss->miniNN_z_latents[k];

    // Layer 0 (12 -> 16)
    alignas(32) int32_t h[MiniNN::CAPTURE_H_DIM];
    for (int j = 0; j < MiniNN::CAPTURE_H_DIM; ++j)
    {
        int32_t sum = cap_b0[j];
        for (int i = 0; i < MiniNN::CAPTURE_IN_DIM; ++i)
            sum += cap_w0[j][i] * x[i];
        h[j] = std::clamp((sum + 32) >> 6, 0, 127);
    }

    // Layer 1 (16 -> 1)
    int32_t sum = cap_b1[0];
    for (int i = 0; i < MiniNN::CAPTURE_H_DIM; ++i)
        sum += cap_w1[0][i] * h[i];

    // Convert scale 4096 to score units [-1200, 1200]
    int score = (sum * 1200 + 2048) >> 12;
    return std::clamp(score, -1200, 1200);
}

int MiniNNModel::evaluate_lmr(
    const Position& pos,
    Move m,
    int moveCount,
    const Search::Stack* ss
) const {
    if (!loaded.load(std::memory_order_relaxed) || !ss)
        return 0;

    Piece pc = pos.moved_piece(m);
    PieceType pt = type_of(pc);
    bool is_capture = pos.capture_stage(m);
    Piece capturedPiece = pos.piece_on(m.to_sq());

    // 8 move features + 8 position latents = 16 inputs (all at scale 64)
    int8_t x[MiniNN::LMR_IN_DIM];
    x[0] = int8_t(std::clamp(int(ss->statScore) * 64 / 2000, -127, 127));
    x[1] = int8_t(std::clamp((moveCount - 4) * 64 / 8, -127, 127));
    x[2] = is_capture ? 64 : -64;
    x[3] = is_capture ? int8_t(std::clamp(int(PieceValue[capturedPiece]) * 64 / 500, 0, 127)) : 0;
    x[4] = pos.gives_check(m) ? 64 : -64;
    x[5] = (m.type_of() == PROMOTION) ? 64 : -64;
    x[6] = int8_t(std::clamp((int(pt) - 2) * 64 / 2, -127, 127));
    x[7] = ss->ttPv ? 64 : -64;

    // Append 8 position latents (scale 64)
    for (int k = 0; k < MiniNN::NODE_LATENTS; ++k)
        x[8 + k] = ss->miniNN_z_latents[k];

    // Layer 0 (16 -> 16)
    alignas(32) int32_t m0[MiniNN::LMR_H_DIM];
    for (int j = 0; j < MiniNN::LMR_H_DIM; ++j)
    {
        int32_t sum = lmr_b0[j];
        for (int i = 0; i < MiniNN::LMR_IN_DIM; ++i)
            sum += lmr_w0[j][i] * x[i];
        m0[j] = std::clamp((sum + 32) >> 6, 0, 127);
    }

    // Layer 1 (16 -> 1)
    int32_t sum = lmr_b1[0];
    for (int i = 0; i < MiniNN::LMR_H_DIM; ++i)
        sum += lmr_w1[0][i] * m0[i];

    // Output reduction in 1024 fixed-point units (1 ply = 1024)
    int r_nn = (sum + 2) >> 2;
    return r_nn;
}

} // namespace Stockfish
