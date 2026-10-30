// Standalone regression test. Run with tests/advanced_pawn_pairs.sh from src/.
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../src/attacks.h"
#include "../src/movegen.h"
#include "../src/position.h"
#include "../src/nnue/nnue_accumulator.h"
#include "../src/nnue/nnue_feature_transformer.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;
using Pair = Features::AdvancedPawnPairs;

namespace {
struct Fixture {
    FeatureTransformer& featureTransformer;
};
int checked = 0;

void compare(const Position&           pos,
             AccumulatorStack&         stack,
             const FeatureTransformer& ft,
             AccumulatorCaches&        cache) {
    stack.evaluate(pos, ft, cache);
    auto fresh = std::make_unique<AccumulatorStack>();
    fresh->reset();
    fresh->evaluate(pos, ft, cache);
    for (Color c : {WHITE, BLACK})
        if (stack.latest().accumulation[c] != fresh->latest().accumulation[c])
        {
            std::cerr << "Incremental mismatch: " << pos.fen() << '\n';
            std::abort();
        }
    ++checked;
}

void walk(Position&                 pos,
          AccumulatorStack&         stack,
          const FeatureTransformer& ft,
          AccumulatorCaches&        cache,
          std::mt19937&             rng,
          int                       depth,
          bool                      lazy) {
    if (!lazy || depth == 0)
        compare(pos, stack, ft, cache);
    if (!depth)
        return;
    std::vector<Move> moves;
    for (Move m : MoveList<ALL>(pos))
        if (pos.legal(m))
            moves.push_back(m);
    std::shuffle(moves.begin(), moves.end(), rng);
    for (Move m : moves)
    {
        StateInfo state;
        auto&     dirty = stack.push(pos);
        pos.do_move(m, state, pos.gives_check(m), dirty);
        for (Color c : {WHITE, BLACK})
        {
            // Stay in the child's orientation for the before/after set difference.
            Pair::IndexList after, removed, added;
            Pair::append_active_indices(c, pos, after);
            Pair::append_changed_indices(c, pos.square<KING>(c), dirty.dirtyAdvancedPawnPairs,
                                         removed, added);
            std::set<u16> reconstructed(after.begin(), after.end());
            for (u16 i : added)
                reconstructed.erase(i);
            for (u16 i : removed)
                reconstructed.insert(i);
            // Independent snapshot-to-index construction for the parent geometry.
            std::set<u16> expected;
            const auto&   old = dirty.dirtyAdvancedPawnPairs.before;
            for (Color pc : {WHITE, BLACK})
                for (Square pawn = SQ_A1; pawn <= SQ_H8; ++pawn)
                    if (old.pawns[pc] & (1ULL << pawn))
                        for (int j = 0; j < old.count; ++j)
                            expected.insert(Pair::make_index(c, pc, pawn, old.pieces[j],
                                                             old.squares[j], pos.square<KING>(c)));
            assert(reconstructed == expected);
        }
        walk(pos, stack, ft, cache, rng, depth - 1, lazy);
        pos.undo_move(m);
        stack.pop();
        if (!lazy)
            compare(pos, stack, ft, cache);
        // Follow one random continuation; main covers every legal root move.
        break;
    }
}
}

int main() {
    Attacks::init();
    Position::init();
    auto ft = std::make_unique<FeatureTransformer>();
    ft->biases.fill(0);
    ft->weights.fill(0);
    ft->threatAndPpWeights.fill(0);
    for (usize i = 0; i < ft->advancedPairWeights.size(); ++i)
        ft->advancedPairWeights[i] = i8((i * 17 + i / 1024 * 11) % 31 - 15);
    ft->permute_weights();
    auto                           cache = std::make_unique<AccumulatorCaches>(Fixture{*ft});
    std::mt19937                   rng(301026);
    const std::vector<std::string> fens = {"4k3/8/2P5/8/8/8/8/R3K3 w - - 0 1",
                                           "4k3/3r4/2P5/8/8/8/8/RNB1K3 w - - 0 1",
                                           "4k2r/2P5/8/8/8/8/8/RN2K3 w - - 0 1",
                                           "4k3/8/8/8/8/2p5/8/R3K2r b - - 0 1",
                                           "4k3/8/8/2Pp4/8/8/8/R3K3 w - d6 0 1",
                                           "4k3/8/2P5/8/8/8/8/R3K2R w KQ - 0 1",
                                           "4k3/8/2P5/8/8/8/8/3K3r w - - 0 1",
                                           "4k3/8/8/2P5/8/8/8/R3K3 w - - 0 1",
                                           "1n2k2r/ppp1pppp/2P5/8/8/3p4/PP1PPPPP/R3K3 w - - 0 1"};
    for (const auto& fen : fens)
        for (bool lazy : {false, true})
        {
            Position  pos;
            StateInfo root;
            pos.set(fen, false, &root);
            auto stack = std::make_unique<AccumulatorStack>();
            stack->reset();
            compare(pos, *stack, *ft, *cache);
            for (Move m : MoveList<ALL>(pos))
            {
                if (!pos.legal(m))
                    continue;
                StateInfo child;
                auto&     dirty = stack->push(pos);
                pos.do_move(m, child, pos.gives_check(m), dirty);
                walk(pos, *stack, *ft, *cache, rng, 3, lazy);
                pos.undo_move(m);
                stack->pop();
                compare(pos, *stack, *ft, *cache);
            }
        }
    std::cout << "Checked " << checked << " incremental/full accumulator comparisons\n";
}
