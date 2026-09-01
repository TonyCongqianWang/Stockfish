/*
  Mini-NN Engine Implementation (Version 4):
  - Dynamic Handcrafted Quiet Terms Weighting (Scale 256, int16_t)
  - evaluate_lmr (Search Depth Reductions)
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
}

bool MiniNNModel::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
        return false;

    uint32_t header[8];
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!file || header[0] != MiniNN::MAGIC || header[1] != MiniNN::VERSION)
        return false;

    // Node Network: fc0 (16 -> 32), fc1 (32 -> 32), fc2 (32 -> 18)
    file.read(reinterpret_cast<char*>(node_b0), sizeof(node_b0));
    file.read(reinterpret_cast<char*>(node_w0), sizeof(node_w0));

    file.read(reinterpret_cast<char*>(node_b1), sizeof(node_b1));
    file.read(reinterpret_cast<char*>(node_w1), sizeof(node_w1));

    file.read(reinterpret_cast<char*>(node_b2), sizeof(node_b2));
    file.read(reinterpret_cast<char*>(node_w2), sizeof(node_w2));

    if (!file)
        return false;

    loaded.store(true, std::memory_order_release);
    return true;
}

void MiniNNModel::extract_node_features(
    const Position& pos,
    const Search::Stack* ss,
    bool improving,
    bool cutNode,
    bool pvNode,
    int8_t out_u[MiniNN::NODE_IN_DIM]
) {
    Color us = pos.side_to_move();
    Color them = ~us;

    int npm_us = pos.non_pawn_material(us);
    int npm_them = pos.non_pawn_material(them);
    int pawns_us = pos.count<PAWN>(us);
    int pawns_them = pos.count<PAWN>(them);

    int num_pinned = popcount(pos.blockers_for_king(us));

    out_u[0]  = int8_t(std::clamp((npm_us - 2000) * 64 / 1000, -127, 127));
    out_u[1]  = int8_t(std::clamp((npm_them - 2000) * 64 / 1000, -127, 127));
    out_u[2]  = int8_t(std::clamp((pawns_us - 4) * 64 / 2, -127, 127));
    out_u[3]  = int8_t(std::clamp((pawns_them - 4) * 64 / 2, -127, 127));
    out_u[4]  = cutNode ? 64 : -64;
    out_u[5]  = int8_t(std::clamp(num_pinned * 64 / 4, -127, 127));
    out_u[6]  = pvNode ? 64 : -64;
    out_u[7]  = ss ? (ss->ttPv ? 64 : -64) : -64;
    out_u[8]  = ss ? int8_t(std::clamp((int(ss->ply) - 24) * 64 / 16, -127, 127)) : 0;
    out_u[9]  = improving ? 64 : -64;
    out_u[10] = ss ? (ss->ttHit ? 64 : -64) : -64;
    out_u[11] = ss ? (ss->ttPv ? 64 : -64) : -64;
    out_u[12] = ss ? int8_t(std::clamp(int(ss->staticEval) * 64 / 500, -127, 127)) : 0;
    out_u[13] = ss ? int8_t(std::clamp(int((ss - 1)->statScore) * 64 / 2000, -127, 127)) : 0;
    out_u[14] = ss ? int8_t(std::clamp((ss->cutoffCnt - 1) * 64 / 2, -127, 127)) : 0;
    out_u[15] = (npm_us + npm_them < 3000) ? 64 : -64;
}

void MiniNNModel::extract_quiet_features(
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
) {
    (void)ss;
    Color us = pos.side_to_move();
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = pos.moved_piece(m);
    PieceType pt = type_of(pc);

    out_x[0] = mainHistory ? int8_t(std::clamp(int((*mainHistory)[us][m.raw()]) * 64 / 16384, -127, 127)) : 0;
    out_x[1] = sharedHistory ? int8_t(std::clamp(int(sharedHistory->pawn_entry(pos)[pc][to]) * 64 / 16384, -127, 127)) : 0;
    out_x[2] = continuationHistory && continuationHistory[0] ? int8_t(std::clamp(int((*continuationHistory[0])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    out_x[3] = continuationHistory && continuationHistory[1] ? int8_t(std::clamp(int((*continuationHistory[1])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    out_x[4] = continuationHistory && continuationHistory[2] ? int8_t(std::clamp(int((*continuationHistory[2])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    out_x[5] = continuationHistory && continuationHistory[3] ? int8_t(std::clamp(int((*continuationHistory[3])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    out_x[6] = continuationHistory && continuationHistory[5] ? int8_t(std::clamp(int((*continuationHistory[5])[pc][to]) * 64 / 16384, -127, 127)) : 0;
    out_x[7] = ((pos.check_squares(pt) & to) && pos.see_ge(m, -75)) ? 64 : -64;
    int v = threatByLesser ? 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to)) * PieceValue[pt] : 0;
    out_x[8] = int8_t(std::clamp(v * 64 / 18000, -127, 127));
    out_x[9] = (ply < LOW_PLY_HISTORY_SIZE && lowPlyHistory) ? int8_t(std::clamp(int((*lowPlyHistory)[ply][m.raw()] / (1 + ply)) * 64 / 16384, -127, 127)) : 0;
}

void MiniNNModel::extract_lmr_features(
    Move m,
    Piece movedPiece,
    bool is_capture,
    Piece capturedPiece,
    bool givesCheck,
    int moveCount,
    const Search::Stack* ss,
    int8_t out_x[MiniNN::LMR_IN_DIM]
) {
    (void)m;
    (void)movedPiece;
    (void)capturedPiece;
    // 6 LMR terms for dynamic residual computation (Scale 64)
    out_x[0] = 64; // Constant bias term (1.0)
    out_x[1] = int8_t(std::clamp((moveCount - 4) * 64 / 8, -127, 127)); // Rank slope
    out_x[2] = ss ? int8_t(std::clamp(int(ss->statScore) * 64 / 2048, -127, 127)) : 0; // History/statScore
    out_x[3] = is_capture ? 64 : 0; // Capture indicator
    out_x[4] = givesCheck ? 64 : 0; // Check indicator
    out_x[5] = !is_capture ? 64 : 0; // Quiet indicator
}

void MiniNNModel::evaluate_node(
    const Position& pos,
    Search::Stack* ss,
    bool improving,
    bool cutNode,
    bool pvNode
) const {
    if (!ss)
        return;

    if (!loaded.load(std::memory_order_relaxed))
    {
        std::memset(ss->miniNN_w_mp, 0, sizeof(ss->miniNN_w_mp));
        std::memset(ss->miniNN_w_lmr, 0, sizeof(ss->miniNN_w_lmr));
        return;
    }

    bool mp_on = use_mp.load(std::memory_order_relaxed);
    bool lmr_on = use_lmr.load(std::memory_order_relaxed);

    if (!mp_on && !lmr_on)
    {
        std::memset(ss->miniNN_w_mp, 0, sizeof(ss->miniNN_w_mp));
        std::memset(ss->miniNN_w_lmr, 0, sizeof(ss->miniNN_w_lmr));
        return;
    }

    // 16 node inputs in [-127, 127]
    int8_t u[MiniNN::NODE_IN_DIM];
    extract_node_features(pos, ss, improving, cutNode, pvNode, u);

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

    // Layer 2 (32 -> 18)
    // 0..9: delta_w_mp residuals for 10 quiet terms (Scale 256: range [-512, 512] => [-2.0x, +2.0x])
    if (mp_on)
    {
        for (int k = 0; k < MiniNN::QUIET_TERMS; ++k)
        {
            int32_t sum = node_b2[k];
            for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
                sum += node_w2[k][i] * h1[i];
            ss->miniNN_w_mp[k] = int16_t(std::clamp((sum + 8) >> 4, -512, 512));
        }
    }
    else
    {
        std::memset(ss->miniNN_w_mp, 0, sizeof(ss->miniNN_w_mp));
    }

    // 10..15: delta_w_lmr residuals for 6 LMR terms (Scale 64: range [-128, 128] => [-2.0, +2.0 plies])
    if (lmr_on)
    {
        for (int k = 0; k < MiniNN::LMR_TERMS; ++k)
        {
            int32_t sum = node_b2[MiniNN::QUIET_TERMS + k];
            for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
                sum += node_w2[MiniNN::QUIET_TERMS + k][i] * h1[i];
            ss->miniNN_w_lmr[k] = int16_t(std::clamp((sum + 32) >> 6, -128, 128));
        }
    }
    else
    {
        std::memset(ss->miniNN_w_lmr, 0, sizeof(ss->miniNN_w_lmr));
    }

    // 18: log_tau_mp
    int32_t sum_tau_mp = node_b2[18];
    for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
        sum_tau_mp += node_w2[18][i] * h1[i];
    int log_tau_mp = std::clamp(sum_tau_mp / 4096, -64, 64);
    ss->miniNN_inv_tau_mp = std::clamp(1024 - (log_tau_mp * 16), 256, 4096);

    // 19: log_tau_lmr
    int32_t sum_tau_lmr = node_b2[19];
    for (int i = 0; i < MiniNN::NODE_H_DIM; ++i)
        sum_tau_lmr += node_w2[19][i] * h1[i];
    int log_tau_lmr = std::clamp(sum_tau_lmr / 4096, -64, 64);
    ss->miniNN_inv_tau_lmr = std::clamp(1024 - (log_tau_lmr * 16), 256, 4096);
}

} // namespace Stockfish
