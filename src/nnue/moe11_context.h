// MOE11 context schema. Keep in sync with Stockfish's nnue/moe11_context.h.
#ifndef MOE11_CONTEXT_H
#define MOE11_CONTEXT_H

#include <algorithm>
#include <array>
#include <cstdint>

namespace Moe11 {

constexpr int           MaterialStates = 240;
constexpr int           KingStates     = 8;
constexpr int           WingStates     = 5;
constexpr int           ContextCount   = MaterialStates * MaterialStates * WingStates * KingStates;
constexpr int           ExpertCount    = 64;
constexpr std::uint32_t SchemaHash     = 0x4D4F453Bu;

constexpr int relative_pawns(int pieces, int pawns) {
    const int officers = pieces - pawns;
    return pawns == 0 ? 0 : pawns <= officers ? 1 : pawns <= officers + 2 ? 2 : 3;
}

constexpr auto MaterialIds = [] {
    struct Table {
        std::array<std::array<std::array<int, 6>, 4>, 16> ids{};
        int                                               count = 0;
    } table;
    for (int t = 0; t < 16; ++t)
    {
        for (int p = 0; p <= std::min(8, t); ++p)
            for (int h = 0; h < 6; ++h)
                if (constexpr int minimum[] = {1, 3, 1, 3, 0, 1};
                    t - p >= minimum[h] && (h != 4 || t == p))
                    table.ids[t][relative_pawns(t, p)][h] = -1;
        for (auto& row : table.ids[t])
            for (int& id : row)
                if (id == -1)
                    id = table.count++;
    }
    return table;
}();

static_assert(MaterialIds.count == MaterialStates);

struct Side {
    int material;
    int wings;
};

inline Side
side(int pieces, int pawnCount, int rooks, int queens, std::uint64_t bishops, std::uint64_t pawns) {
    const int               total   = std::clamp(pieces - 1, 0, 15);
    const int               p       = std::clamp(pawnCount, 0, std::min(8, total));
    constexpr std::uint64_t Dark    = 0x55AA55AA55AA55AAULL;
    constexpr std::uint64_t Left    = 0x0F0F0F0F0F0F0F0FULL;
    const bool              pair    = (bishops & Dark) && (bishops & ~Dark);
    const bool              support = total - p - rooks - queens + pair > rooks + 2 * queens;
    const int               heavy   = 2 * (queens ? 0 : rooks ? 1 : 2) + support;
    return {MaterialIds.ids[total][relative_pawns(total, p)][heavy],
            bool(pawns & Left) + 2 * bool(pawns & ~Left)};
}

inline int context(Side own, Side enemy, int ownKing, int enemyKing, bool blackToMove) {
    const int us = own.wings, them = enemy.wings;
    const int wing = us == 3 || them == 3     ? (us == them ? 0
                                                 : us == 3  ? 1
                                                            : 2)
                   : us && them && us != them ? 3
                                              : 4;

    // Orient each king from its own perspective. Files stay in board coordinates.
    ownKing ^= 56 * blackToMove;
    enemyKing ^= 56 * !blackToMove;
    const bool opposite = (ownKing % 8 < 4) != (enemyKing % 8 < 4);
    const int  king     = 4 * (ownKing / 8 >= 2) + 2 * (ownKing / 8 > enemyKing / 8) + opposite;
    return ((own.material * MaterialStates + enemy.material) * WingStates + wing) * KingStates
         + king;
}

}  // namespace Moe11

#endif
