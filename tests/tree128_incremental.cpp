// Link against the matching engine's objects except main.o.
// Usage: tree128_incremental matching.nnue [roundtrip.nnue]
#include "../src/attacks.h"
#include "../src/position.h"
#include "../src/movegen.h"
#include "../src/nnue/network.h"
#include "../src/nnue/nnue_misc.h"
#include "../src/nnue/nnue_accumulator.h"
#include <iostream>
#include <memory>
#include <random>

using namespace Stockfish;
int main(int argc, char** argv) {
    if (argc<2) return 2;
    Attacks::init();
    auto net = std::make_unique<Eval::NNUE::Network>();
    Eval::NNUE::EvalFile file;
    net->load("", argv[1], file);
    if (!file.current) return 3;
    auto incremental = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto fresh = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto cache = std::make_unique<Eval::NNUE::AccumulatorCaches>(*net);
    auto freshCache = std::make_unique<Eval::NNUE::AccumulatorCaches>(*net);
    std::mt19937 rng(128);
    const char* starts[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "7k/P7/8/8/8/8/7p/K7 w - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        "7k/8/8/8/8/BB1B4/NNN5/KRRQQ3 w - - 0 1"
    };
    int checked=0;
    for (int game=0; game<48; ++game) {
        Position pos;
        if (auto error = pos.set(starts[game%4], false)) { std::cerr << error->what(); return 4; }
        incremental->reset();
        for (int ply=0; ply<96; ++ply) {
            fresh->reset();
            freshCache->clear(*net);
            Position rebuilt;
            if (rebuilt.set(pos.fen(), false)) return 5;
            auto a=net->evaluate(pos, *incremental, *cache);
            auto b=net->evaluate(rebuilt, *fresh, *freshCache);
            if (a!=b) { std::cerr << "incremental mismatch " << pos.fen() << ' ' << a << ' ' << b << '\n'; return 1; }
            ++checked;
            MoveList<ALL> moves(pos);
            if (!moves.size()) break;
            Move move=moves.begin()[rng()%moves.size()];
            Dirties& dirty=incremental->push(pos);
            pos.do_move(move, pos.gives_check(move), dirty);
            // Exercise pop/undo as well as accumulation forward and refresh.
            if (ply%7==0) {
                (void)net->evaluate(pos, *incremental, *cache);
                pos.undo_move(move);
                incremental->pop();
                if (net->evaluate(pos, *incremental, *cache)!=a) return 6;
                Dirties& again=incremental->push(pos);
                pos.do_move(move, pos.gives_check(move), again);
            }
        }
    }
    if (argc>2 && !net->save(file,std::filesystem::path(argv[2]))) return 7;
    std::cout << "Verified " << checked << " incremental/fresh positions and undo branches\n";
}
