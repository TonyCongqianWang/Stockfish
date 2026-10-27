/* Copyright (C) 2026 The Stockfish developers. GPL-3.0-or-later.
   Training-free feature, incremental accumulator, and activation regression test. */
#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "attacks.h"
#include "bitboard.h"
#include "movegen.h"
#include "position.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/nnue_feature_transformer.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;

namespace {
long long checks = 0;
void      require(bool condition, const std::string& message) {
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
int weight(int family, int row, int lane) { return (row * 17 + lane * 3 + family * 11) % 7 - 3; }

std::vector<Move> legal(const Position& pos) {
    std::vector<Move> out;
    if (pos.checkers())
    {
        for (Move m : MoveList<EVASIONS>(pos))
            if (pos.legal(m))
                out.push_back(m);
    }
    else
    {
        for (Move m : MoveList<ALL>(pos))
            if (pos.legal(m))
                out.push_back(m);
    }
    return out;
}

void verify(Position&           pos,
            AccumulatorStack&   stack,
            FeatureTransformer& ft,
            AccumulatorCaches&  cache) {
    stack.evaluate(pos, ft, cache);
    std::array<std::array<int, L1>, 2> support{};
    for (Color c : {WHITE, BLACK})
        for (bool enemy : {false, true})
        {
            SupportFeatureSet::IndexList active;
            SupportFeatureSet::append_active_indices(c, enemy, pos, active);
            const int          width  = enemy ? EnemySupportDimensions : OwnSupportDimensions;
            const int          offset = enemy ? OwnSupportDimensions : 0;
            const Color        owner  = enemy ? ~c : c;
            std::array<int, 7> reference{};
            for (int j = 0; j < 7; ++j)
                for (auto i : active)
                    reference[j] += weight(enemy ? 3 : 0, i, j);
            for (int j = 0; j < width; ++j)
            {
                int expected = reference[j % 7];
                if (stack.latest().supportAccumulation[owner][offset + j] != expected)
                    require(false, "support accumulator mismatch: " + pos.fen());
                const int operand = j / (width / 2), k = j % (width / 2);
                support[c][operand * L1 / 2 + (enemy ? OwnSupportDimensions / 2 : 0) + k] =
                  expected;
            }
        }

    std::array<std::array<int, 7>, 2> unrestricted{};
    for (Color c : {WHITE, BLACK})
    {
        auto& expected = unrestricted[c];
        for (int j = 0; j < 7; ++j)
            expected[j] = 64 + j;
        ThreatFeatureSet::IndexList indices;
        ThreatFeatureSet::append_active_indices(c, pos, indices);
        PairFeatureSet::append_active_indices(c, pos, indices);
        for (auto i : indices)
        {
            require(i < FeatureTransformer::ThreatAndPpDimensions, "global index overflow");
            for (int j = 0; j < 7; ++j)
                expected[j] += weight(1, i, j);
        }
        Bitboard occupied = pos.pieces();
        while (occupied)
        {
            Square square = pop_lsb(occupied);
            auto i = PSQFeatureSet::make_index(c, square, pos.piece_on(square), pos.square<KING>(c),
                                               bool(pos.pieces(~c, QUEEN)));
            for (int j = 0; j < 7; ++j)
                expected[j] += weight(2, i, j);
        }
        auto actual = stack.latest().accumulation[c];
        permute<16>(actual, FeatureTransformer::InversePackusEpi16Order);
        for (IndexType j = 0; j < L1; ++j)
            if (actual[j] != expected[j % 7])
                require(false, "unrestricted accumulator mismatch: " + pos.fen());
    }
    alignas(CacheLineSize) std::array<TransformedFeatureType, L1> output{};
    NNZInfo<L1>                                                   nnz{};
    ft.transform(pos, stack, cache, output.data(), nnz);
    for (int perspective = 0; perspective < 2; ++perspective)
    {
        Color c = perspective == 0 ? pos.side_to_move() : ~pos.side_to_move();
        for (IndexType j = 0; j < L1 / 2; ++j)
        {
            auto value = [&](IndexType k) {
                return std::clamp(unrestricted[c][k % 7] + support[c][k], 0, 255);
            };
            int expected = value(j) * value(j + L1 / 2) / 512;
            if (output[perspective * (L1 / 2) + j] != expected)
                require(false, "activation mismatch: " + pos.fen());
        }
    }
    ++checks;
}

template<typename List>
void json_list(const List& values, int subtract = 0) {
    std::vector<int> sorted;
    for (auto i : values)
        sorted.push_back(int(i) - subtract);
    std::sort(sorted.begin(), sorted.end());
    std::cout << '[';
    for (std::size_t i = 0; i < sorted.size(); ++i)
    {
        if (i)
            std::cout << ',';
        std::cout << sorted[i];
    }
    std::cout << ']';
}
void dump(const Position& pos) {
    std::cout << "{";
    for (Color c : {WHITE, BLACK})
        for (bool enemy : {false, true})
        {
            SupportFeatureSet::IndexList active;
            SupportFeatureSet::append_active_indices(c, enemy, pos, active);
            if (c != WHITE || enemy)
                std::cout << ',';
            std::cout << "\"" << (enemy ? "enemy_" : "own_") << (c == WHITE ? "w" : "b") << "\":";
            json_list(active);
        }
    for (Color c : {WHITE, BLACK})
    {
        ThreatFeatureSet::IndexList attacks, pawns;
        ThreatFeatureSet::append_active_indices(c, pos, attacks);
        PairFeatureSet::append_active_indices(c, pos, pawns);
        std::cout << (c == WHITE ? ",\"attack_w\":" : ",\"attack_b\":");
        json_list(attacks);
        std::cout << (c == WHITE ? ",\"pawn_w\":" : ",\"pawn_b\":");
        json_list(pawns, PairFeatureSet::IndexBase);
    }
    std::cout << "}\n";
}

void indices() {
    for (int f = 0; f < 8; ++f)
    {
        std::array<bool, 60> seen{};
        for (int a = 0; a < 12; ++a)
            for (int b = a + 1; b < 12; ++b)
            {
                if (a % 6 == b % 6)
                    continue;
                int i = PairFeatureSet::file_pair_index(WHITE, f, a, b) - PairFeatureSet::IndexBase
                      - PairFeatureSet::FrontierDimensions - 60 * f;
                require(i >= 0 && i < 60 && !seen[i], "file-pair indexing collision");
                seen[i] = true;
                int ar = (1 - a / 6) * 6 + 5 - a % 6, br = (1 - b / 6) * 6 + 5 - b % 6;
                require(PairFeatureSet::file_pair_index(BLACK, f, ar, br)
                          == PairFeatureSet::file_pair_index(WHITE, f, a, b),
                        "file-pair perspective mismatch");
            }
        require(std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }),
                "missing file-pair state");
    }
    // Every allocated frontier state, including empty and impossible collision
    // states, has the prescribed color/rank involution in the pure state map.
    auto flip = [](int s) {
        int w = s / 7, b = s % 7;
        return 7 * (b ? 7 - b : 0) + (w ? 7 - w : 0);
    };
    for (int a = 0; a < 49; ++a)
        for (int b = 0; b < 49; ++b)
            require(flip(flip(a)) == a && flip(flip(b)) == b, "frontier involution mismatch");
    for (int ca = 0; ca < 2; ++ca)
        for (int ta = 1; ta <= 5; ++ta)
            for (int cb = 0; cb < 2; ++cb)
                for (int tb = 1; tb <= 5; ++tb)
                    for (int f = 0; f < 64; ++f)
                    {
                        if (ta == PAWN && (f < 8 || f >= 56))
                            continue;
                        Bitboard targets = ta == PAWN ? Attacks::PseudoAttacks[ca][f]
                                                      : Attacks::PseudoAttacks[ta][f];
                        while (targets)
                        {
                            Square to = pop_lsb(targets);
                            Piece  a  = make_piece(Color(ca), PieceType(ta)),
                                   b  = make_piece(Color(cb), PieceType(tb));
                            if (ca == cb)
                            {
                                auto i =
                                  SupportFeatureSet::make_index(Color(ca), a, Square(f), to, b);
                                auto t = SupportFeatureSet::make_index(
                                  Color(1 - ca), Piece(a ^ 8), Square(f ^ 56), Square(int(to) ^ 56),
                                  Piece(b ^ 8));
                                require(i == t && i <= SupportFeatureSet::Dimensions,
                                        "support perspective mismatch");
                            }
                            else
                            {
                                auto i = ThreatFeatureSet::make_index(WHITE, a, Square(f), to, b);
                                auto t =
                                  ThreatFeatureSet::make_index(BLACK, Piece(a ^ 8), Square(f ^ 56),
                                                               Square(int(to) ^ 56), Piece(b ^ 8));
                                require(i == t && i <= ThreatFeatureSet::Dimensions,
                                        "attack perspective mismatch");
                            }
                        }
                    }
}
}  // namespace

int main(int argc, char** argv) {
    Attacks::init();
    Position::init();
    indices();
    bool                     dumping = argc > 1 && std::string(argv[1]) == "--dump";
    int                      pathArg = dumping ? 2 : 1;
    std::vector<std::string> fens    = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/ppp2ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPP2PPP/R3K2R w KQkq - 0 8",
      "4k3/8/8/4Pp2/8/8/8/4K3 w - f6 0 1", "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
      "4k3/8/2p5/2pP4/2P5/2P5/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
      "4k3/8/8/8/8/8/4q3/R2K3R w - - 0 1",
      // Independent king mirrors with active own and enemy pawn/knight support.
      "3k4/8/3p4/2n5/2N5/3P4/8/3K4 w - - 0 1", "4k3/8/3p4/2n5/2N5/3P4/8/3K4 w - - 0 1",
      "3k4/8/3p4/2n5/2N5/3P4/8/4K3 w - - 0 1", "4k3/8/3p4/2n5/2N5/3P4/8/4K3 w - - 0 1",
      // Chess960: king/rook swap squares, and castling with a stationary king.
      "4k3/8/8/8/8/8/8/5KR1 w G - 0 1 chess960", "4k3/8/8/8/8/8/8/1RK5 w B - 0 1 chess960"};
    if (argc > pathArg)
    {
        fens.clear();
        std::ifstream in(argv[pathArg]);
        require(bool(in), "cannot read positions");
        std::string line;
        while (std::getline(in, line))
            if (!line.empty())
                fens.push_back(line);
    }
    if (dumping)
    {
        for (const auto& fen : fens)
        {
            Position  pos;
            StateInfo state{};
            require(!pos.set(fen, fen.find("chess960") != std::string::npos, &state),
                    "invalid FEN");
            dump(pos);
        }
        return 0;
    }
    auto ft = std::make_unique<FeatureTransformer>();
    for (IndexType j = 0; j < L1; ++j)
        ft->biases[j] = 64 + j % 7;
    for (std::size_t i = 0; i < ft->supportWeights.size(); ++i)
        ft->supportWeights[i] = weight(0, i / OwnSupportDimensions, i % OwnSupportDimensions);
    for (std::size_t i = 0; i < ft->enemySupportWeights.size(); ++i)
        ft->enemySupportWeights[i] =
          weight(3, i / EnemySupportDimensions, i % EnemySupportDimensions);
    for (std::size_t i = 0; i < ft->threatAndPpWeights.size(); ++i)
        ft->threatAndPpWeights[i] = weight(1, i / L1, i % L1);
    for (std::size_t i = 0; i < ft->weights.size(); ++i)
        ft->weights[i] = weight(2, i / L1, i % L1);
    ft->permute_weights();
    auto stack = std::make_unique<AccumulatorStack>();
    struct TestNetwork {
        FeatureTransformer& featureTransformer;
    } network{*ft};
    auto      cache = std::make_unique<AccumulatorCaches>(network);
    long long edges = 0;
    for (std::size_t i = 0; i < fens.size(); ++i)
    {
        Position  pos;
        StateInfo root{};
        require(!pos.set(fens[i], fens[i].find("chess960") != std::string::npos, &root),
                "invalid FEN");
        stack->reset();
        if (i % 2 == 0)
            verify(pos, *stack, *ft, *cache);
        for (Move m : legal(pos))
        {
            StateInfo child{};
            pos.do_move(m, child, pos.gives_check(m), stack->push());
            verify(pos, *stack, *ft, *cache);
            ++edges;
            pos.undo_move(m);
            stack->pop();
        }
        verify(pos, *stack, *ft, *cache);
        if (!pos.checkers())
        {
            StateInfo nullState{};
            pos.do_null_move(nullState);
            verify(pos, *stack, *ft, *cache);
            pos.undo_null_move();
        }
        // Deferred updates, backward filling across king/queen refresh barriers,
        // and repeated pop/reuse of the same accumulator slots.
        if (i < 64)
        {
            std::deque<StateInfo> states;
            std::vector<Move>     moves;
            for (int ply = 0; ply < 20; ++ply)
            {
                auto available = legal(pos);
                if (available.empty())
                    break;
                Move m = available[(i * 17 + ply * 11) % available.size()];
                states.emplace_back();
                pos.do_move(m, states.back(), pos.gives_check(m), stack->push());
                moves.push_back(m);
                if (ply % 4 == 3)
                    verify(pos, *stack, *ft, *cache);
            }
            while (!moves.empty())
            {
                pos.undo_move(moves.back());
                moves.pop_back();
                stack->pop();
                verify(pos, *stack, *ft, *cache);
            }
        }
        if ((i + 1) % 1000 == 0)
            std::cerr << "positions=" << i + 1 << " edges=" << edges << " checks=" << checks
                      << '\n';
    }
    std::cout << "PASS positions=" << fens.size() << " legal_edges=" << edges
              << " accumulator_checks=" << checks << '\n';
}
