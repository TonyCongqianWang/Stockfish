// Standalone correctness/performance harness. No search or bench command.
#include <chrono>
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
using Composition = Features::Composition;

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
    if (argc > 1 && std::string(argv[1]) == "features")
    {
        std::string fen;
        while (std::getline(std::cin, fen))
        {
            Position  pos;
            StateInfo st;
            require(!pos.set(fen, false, &st), "invalid FEN");
            auto counts = Composition::counts(pos);
            for (Color c : {WHITE, BLACK})
            {
                int b = Composition::bucket(counts, c), row = Composition::Offsets[b];
                std::cout << b;
                for (int t = 0; t < 10; ++t)
                {
                    int n = counts[(t + 5 * c) % 10];
                    for (int i = 0; i < n; ++i)
                        std::cout << ' ' << row;
                    row += n != 0;
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
    for (auto& w : ft.materialWeights)
        w = int(rng() % 17) - 8;
    ft.biases.fill(100);
    ft.permute_weights();
    auto     cache          = std::make_unique<AccumulatorCaches>(*fixture);
    auto     referenceCache = std::make_unique<AccumulatorCaches>(*fixture);
    auto     stack          = std::make_unique<AccumulatorStack>();
    auto     reference      = std::make_unique<AccumulatorStack>();
    unsigned checks = 0, quiet = 0, capture = 0, promotion = 0, ep = 0, castle = 0;
    unsigned presence = 0, sameBucket = 0, king = 0, nulls = 0;
    auto     check = [&](Position& pos) {
        stack->evaluate(pos, ft, *cache);
        reference->reset();
        referenceCache->clear(*fixture);
        reference->evaluate(pos, ft, *referenceCache);
        require(stack->latest().accumulation == reference->latest().accumulation,
                    "incremental/full accumulator mismatch");
        require(stack->latest().materialCounts == Composition::counts(pos), "counts mismatch");
        ++checks;
    };
    // Three presence changes without a king/H3 refresh: evaluate only the leaf.
    // Intermediate spatial states retain the root material anchor, rather than
    // constructing each crossed bucket. Undo must reconcile those states on demand.
    {
        Position  pos;
        StateInfo root, children[3];
        require(!pos.set("7k/8/8/8/8/1b6/n1B5/R6K w - - 0 1", false, &root), "lazy FEN");
        stack->reset();
        check(pos);
        const Move moves[] = {Move(SQ_A1, SQ_A2), Move(SQ_B3, SQ_C2), Move(SQ_A2, SQ_C2)};
        const auto updates = Composition::incrementalUpdates;
        for (int i = 0; i < 3; ++i)
        {
            require(pos.legal(moves[i]), "lazy move legality");
            auto& d = stack->push(pos);
            pos.do_move(moves[i], children[i], pos.gives_check(moves[i]), d);
        }
        check(pos);
        require(Composition::incrementalUpdates - updates == 2,
                "lazy chain must reconcile only final material, once per perspective");
        for (int i = 2; i >= 0; --i)
        {
            pos.undo_move(moves[i]);
            stack->pop();
            check(pos);
        }
    }

    const char* fens[] = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/ppp2ppp/2n1bn2/3qp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 1",
      "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
      "4k3/P6r/8/8/8/8/7p/R3K3 w Q - 0 1",
      "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
      "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
      "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
      "8/8/3k4/8/3Kp3/8/8/8 w - - 0 1",
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
            auto      before = Composition::counts(pos);
            StateInfo child;
            auto&     dirties = stack->push(pos);
            pos.do_move(m, child, pos.gives_check(m), dirties);
            auto       after   = Composition::counts(pos);
            const bool changed = before != after;
            require(Composition::changes(dirties.dirtyPiece) == changed,
                    "quiet/change classification");
            require(Composition::changed(before, dirties.dirtyPiece, 1) == after, "forward counts");
            require(Composition::changed(after, dirties.dirtyPiece, -1) == before,
                    "reverse counts");
            quiet += !changed;
            capture += child.capturedPiece != NO_PIECE;
            promotion += m.type_of() == PROMOTION;
            ep += m.type_of() == EN_PASSANT;
            castle += m.type_of() == CASTLING;
            king += type_of(dirties.dirtyPiece.pc) == KING;
            if (changed)
            {
                bool equal =
                  Composition::bucket(before, WHITE) == Composition::bucket(after, WHITE);
                sameBucket += equal;
                presence += !equal;
            }
            const auto updates = Composition::incrementalUpdates;
            check(pos);
            if (!changed)
                require(updates == Composition::incrementalUpdates, "quiet material update");
            pos.undo_move(m);
            stack->pop();
            check(pos);
        }
        // Lazy paths, mixed king refreshes, captures, and undoing computed/uncomputed nodes.
        for (int trial = 0; trial < 12; ++trial)
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
    require(quiet && capture && promotion && ep && castle && presence && sameBucket && king
              && nulls,
            "missing edge coverage");
    std::cout << "ft_bytes=" << sizeof(FeatureTransformer)
              << " accumulator_state_bytes=" << sizeof(AccumulatorState)
              << " finny_entry_bytes=" << sizeof(AccumulatorCaches::Entry) << '\n';
    std::cout << "checks=" << checks << " quiet=" << quiet << " captures=" << capture
              << " promotions=" << promotion << " ep=" << ep << " castling=" << castle
              << " presence=" << presence << " unchanged_bucket=" << sameBucket << " king=" << king
              << " null=" << nulls << '\n';
}
