#include <chrono>
#include <iostream>
#include <memory>
#include "attacks.h"
#include "position.h"
#include "movegen.h"
#include "nnue/nnue_feature_transformer.h"
using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;
struct Fixture {
    FeatureTransformer featureTransformer;
};
int main() {
    Attacks::init();
    Position::init();
    auto  f  = std::make_unique<Fixture>();
    auto& ft = f->featureTransformer;
    ft.weights.fill(1);
    ft.threatAndPpWeights.fill(1);
    ft.biases.fill(50);
#ifdef MATERIAL
    ft.materialWeights.fill(1);
#endif
    ft.permute_weights();
    auto          cache    = std::make_unique<AccumulatorCaches>(*f);
    auto          stack    = std::make_unique<AccumulatorStack>();
    const char*   fens[]   = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                              "r3k2r/ppp2ppp/2n1bn2/3qp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1",
                              "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"};
    volatile long checksum = 0;
    for (auto fen : fens)
    {
        Position  pos;
        StateInfo root;
        if (pos.set(fen, false, &root))
            return 1;
        for (int category = 0; category < 4; ++category)
        {
            Move chosen = Move::none();
            for (Move m : MoveList<ALL>(pos))
            {
                if (!pos.legal(m))
                    continue;
                bool capture = pos.capture(m);
                bool king    = type_of(pos.moved_piece(m)) == KING;
                if ((category == 0 && !capture && !king) || (category == 1 && capture)
                    || (category == 2 && king && !capture))
                {
                    chosen = m;
                    break;
                }
            }
            if (chosen == Move::none() && category != 3)
                continue;
            stack->reset();
            stack->evaluate(pos, ft, *cache);
            auto          start = std::chrono::steady_clock::now();
            constexpr int N     = 60000;
            for (int i = 0; i < N; ++i)
            {
                if (category == 3)
                {
                    stack->evaluate(pos, ft, *cache);
                    checksum += stack->latest().accumulation[WHITE][i % 1024];
                    continue;
                }
                StateInfo child;
                auto&     d = stack->push(pos);
                pos.do_move(chosen, child, pos.gives_check(chosen), d);
                stack->evaluate(pos, ft, *cache);
                checksum += stack->latest().accumulation[WHITE][i % 1024];
                pos.undo_move(chosen);
                stack->pop();
            }
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
            std::cout << category << ' ' << double(ns) / N << '\n';
        }
    }
    std::cerr << "ft_bytes=" << sizeof(FeatureTransformer)
              << " state_bytes=" << sizeof(AccumulatorState)
              << " finny_entry_bytes=" << sizeof(AccumulatorCaches::Entry) << '\n';
    std::cerr << checksum << '\n';
}
