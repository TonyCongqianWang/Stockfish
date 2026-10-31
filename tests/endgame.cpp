// Synthetic accumulator and feature parity checks; no search or bench.
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "../src/bitboard.h"
#include "../src/movegen.h"
#include "../src/position.h"
#include "../src/nnue/nnue_feature_transformer.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;
using Endgame = Features::Endgame;

void require(bool ok, const char* message) {
    if (!ok)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
struct Fixture {
    FeatureTransformer featureTransformer;
};

int main(int argc, char** argv) {
    Attacks::init();
    Position::init();
    if (argc > 1)
    {
        std::string fen;
        while (std::getline(std::cin, fen))
        {
            Position  pos;
            StateInfo st;
            require(!pos.set(fen, false, &st), "invalid FEN");
            const auto board = Endgame::snapshot(pos);
            for (Color p : {WHITE, BLACK})
            {
                for (Color c : {p, ~p})
                {
                    const int cat = Endgame::category(board[c]);
                    if (cat < 0)
                        continue;
                    for (int t = 0; t < 5; ++t)
                    {
                        auto bb = board[c][t];
                        while (bb)
                            std::cout
                              << Endgame::make_index(cat, bool(board[c][0]), t, c != p, pop_lsb(bb),
                                                     Endgame::orientation(p, pos.square<KING>(p)))
                              << ' ';
                    }
                }
                std::cout << '\n';
            }
        }
        return 0;
    }
    auto         fixture = std::make_unique<Fixture>();
    auto&        ft      = fixture->featureTransformer;
    std::mt19937 rng(1729);
    for (auto& w : ft.weights)
        w = int(rng() % 7) - 3;
    for (auto& w : ft.threatAndPpWeights)
        w = int(rng() % 7) - 3;
    for (auto& w : ft.endgameWeights)
        w = int(rng() % 17) - 8;
    ft.biases.fill(100);
    ft.permute_weights();
    auto     cache          = std::make_unique<AccumulatorCaches>(*fixture);
    auto     referenceCache = std::make_unique<AccumulatorCaches>(*fixture);
    auto     stack          = std::make_unique<AccumulatorStack>();
    auto     reference      = std::make_unique<AccumulatorStack>();
    unsigned checks = 0, captures = 0, promotions = 0, ep = 0, castles = 0, kings = 0, nulls = 0;
    auto     check = [&](Position& pos) {
        stack->evaluate(pos, ft, *cache);
        reference->reset();
        referenceCache->clear(*fixture);
        reference->evaluate(pos, ft, *referenceCache);
        require(stack->latest().accumulation == reference->latest().accumulation,
                    "incremental/full accumulator mismatch");
        for (Color p : {WHITE, BLACK})
        {
            const auto& sum = cache->endgameSums[p];
            require(sum.board == Endgame::snapshot(pos), "snapshot mismatch");
            require(sum.accumulation == referenceCache->endgameSums[p].accumulation,
                        "incremental/full endgame sum mismatch");
            // Independent full square-row summation, without cache/delta machinery.
            std::array<i16, L1> expected{};
            for (Color c : {WHITE, BLACK})
            {
                const int cat = Endgame::category(sum.board[c]);
                if (cat < 0)
                    continue;
                for (int t = 0; t < 5; ++t)
                {
                    auto squares = sum.board[c][t];
                    while (squares)
                    {
                        int id = Endgame::make_index(cat, bool(sum.board[c][0]), t, c != p,
                                                         pop_lsb(squares), sum.flip);
                        for (int j = 0; j < int(L1); ++j)
                            expected[j] = i16(expected[j] + ft.endgameWeights[id * L1 + j]);
                    }
                }
            }
            require(sum.accumulation == expected, "endgame sum oracle mismatch");
        }
        ++checks;
    };
    // Three category-changing captures with no intermediate evaluations. Spatial
    // traversals must not reconcile the skipped categories; undo can request them.
    {
        Position  pos;
        StateInfo root, children[3];
        require(!pos.set("7k/8/8/8/8/1b6/n1B5/R6K w - - 0 1", false, &root), "lazy FEN");
        stack->reset();
        check(pos);
        const Move moves[] = {Move(SQ_A1, SQ_A2), Move(SQ_B3, SQ_C2), Move(SQ_A2, SQ_C2)};
        for (int i = 0; i < 3; ++i)
        {
            require(pos.legal(moves[i]), "lazy move legality");
            auto& d = stack->push(pos);
            pos.do_move(moves[i], children[i], pos.gives_check(moves[i]), d);
        }
        const auto reconciliations = Endgame::reconciliations;
        stack->evaluate(pos, ft, *cache);
        require(Endgame::reconciliations - reconciliations == 2,
                "intermediate category reconciliation");
        check(pos);
        for (int i = 2; i >= 0; --i)
        {
            pos.undo_move(moves[i]);
            stack->pop();
            check(pos);
        }
    }
    // A middlegame quiet move is silent; repeating an evaluated position is free
    // of endgame row operations, including on a warm category cache.
    {
        Position  pos;
        StateInfo root, child;
        require(!pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, &root),
                "quiet FEN");
        stack->reset();
        stack->evaluate(pos, ft, *cache);
        const auto rows = Endgame::rowUpdates;
        const Move move(SQ_E2, SQ_E4);
        auto&      d = stack->push(pos);
        pos.do_move(move, child, pos.gives_check(move), d);
        stack->evaluate(pos, ft, *cache);
        stack->evaluate(pos, ft, *cache);
        require(Endgame::rowUpdates == rows, "silent middlegame updated endgame rows");
        pos.undo_move(move);
        stack->pop();
        check(pos);
    }
    const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/ppp2ppp/2n1bn2/3qp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1",
      "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
      "4k3/P6r/8/8/8/8/7p/R3K3 w Q - 0 1",
      "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
      "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
      "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
      "7k/8/8/8/8/1b6/n1B5/R6K w - - 0 1",
      "4k3/pp6/2n5/8/8/5B2/PP6/R3K3 w - - 0 1",
      "4k3/pp6/2b5/8/8/5N2/PP6/R3K3 w - - 0 1",
      "4k3/pp6/2r5/8/8/5N2/PP6/B3K3 w - - 0 1",
      "4k3/pp6/2q5/8/8/5Q2/PP6/4K3 w - - 0 1",
      "4k3/8/2n5/8/8/5B2/8/N3K3 w - - 0 1",
      "4k3/8/2b5/8/8/5R2/8/N3K3 w - - 0 1",
      "4k3/8/2r5/8/8/5R2/8/B3K3 w - - 0 1",
      "4k3/8/2q5/8/8/5Q2/8/4K3 w - - 0 1",
    };
    for (const char* fen : fens)
    {
        Position  pos;
        StateInfo root;
        require(!pos.set(fen, false, &root), "invalid test FEN");
        stack->reset();
        check(pos);
        for (Move m : MoveList<ALL>(pos))
        {
            if (!pos.legal(m))
                continue;
            StateInfo child;
            auto&     d = stack->push(pos);
            pos.do_move(m, child, pos.gives_check(m), d);
            captures += child.capturedPiece != NO_PIECE;
            promotions += m.type_of() == PROMOTION;
            ep += m.type_of() == EN_PASSANT;
            castles += m.type_of() == CASTLING;
            kings += type_of(d.dirtyPiece.pc) == KING;
            check(pos);
            pos.undo_move(m);
            stack->pop();
            check(pos);
        }
        for (int trial = 0; trial < 8; ++trial)
        {
            std::vector<Move> moves;
            StateInfo         states[80];
            for (int ply = 0; ply < 80; ++ply)
            {
                std::vector<Move> legal;
                for (Move m : MoveList<ALL>(pos))
                    if (pos.legal(m))
                        legal.push_back(m);
                if (legal.empty())
                    break;
                Move  m = legal[rng() % legal.size()];
                auto& d = stack->push(pos);
                pos.do_move(m, states[ply], pos.gives_check(m), d);
                moves.push_back(m);
                if (ply % 7 == 6)
                    check(pos);
                if (!pos.checkers() && ply % 13 == 0)
                {
                    StateInfo nullState;
                    pos.do_null_move(nullState);
                    check(pos);
                    pos.undo_null_move();
                    check(pos);
                    ++nulls;
                }
            }
            check(pos);
            for (auto it = moves.rbegin(); it != moves.rend(); ++it)
            {
                pos.undo_move(*it);
                stack->pop();
                if (rng() % 3 == 0)
                    check(pos);
            }
            check(pos);
        }
    }
    // Invalidation must discard all old category sums after network replacement.
    for (auto& w : ft.endgameWeights)
        ++w;
    cache->clear(*fixture);
    stack->reset();
    Position  pos;
    StateInfo root;
    require(!pos.set(fens[8], false, &root), "invalidation FEN");
    check(pos);
    require(captures && promotions && ep && castles && kings && nulls, "missing move coverage");
    std::cout << "checks=" << checks << " captures=" << captures << " promotions=" << promotions
              << " ep=" << ep << " castles=" << castles << " kings=" << kings << " nulls=" << nulls
              << " state_bytes=" << sizeof(AccumulatorState)
              << " category_cache_bytes=" << sizeof(cache->endgames) << '\n';
}
