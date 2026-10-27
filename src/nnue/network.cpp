/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "network.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>
#include <filesystem>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../incbin/incbin.h"

#include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_misc.h"
#include "nnz_helper.h"

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(UNIVERSAL_BINARY) && !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#elif defined(UNIVERSAL_BINARY_MACOS_X86_SLICE)
// Determined at runtime, see universal/nnue_embed.cpp
extern const unsigned char* const gEmbeddedNNUEData;
extern const unsigned int         gEmbeddedNNUESize;
#elif defined(UNIVERSAL_BINARY)
extern const unsigned char gEmbeddedNNUEData[];
extern const unsigned int  gEmbeddedNNUESize;
#else
const unsigned char gEmbeddedNNUEData[1] = {0x0};
const unsigned int  gEmbeddedNNUESize    = 1;
#endif


namespace Stockfish::Eval::NNUE {

namespace fs = std::filesystem;

namespace {

// Legal material states (non-king pieces, pawns, cascading material category).
constexpr auto MAT_ID_TABLE = [] {
    std::array<std::array<std::array<uint16_t, 4>, 9>, 16> table{};
    uint16_t                                               id = 0;
    for (int t = 0; t < 16; ++t)
        for (int p = 0; p <= std::min(8, t); ++p)
            for (int h = 0; h < 4; ++h)
                if (t - p >= (h < 2 ? 1 : h == 2 ? 2 : 0))
                    table[t][p][h] = id++;
    return table;
}();

inline uint16_t get_material_id(const Position& pos, Color c) {
    const int      t = std::clamp(pos.count<ALL_PIECES>(c) - 1, 0, 15);
    const int      p = std::clamp(pos.count<PAWN>(c), 0, std::min(8, t));
    const Bitboard b = pos.pieces(c, BISHOP);
    const int      h = t - p >= 1 && pos.pieces(c, QUEEN)                                        ? 0
                     : t - p >= 1 && pos.pieces(c, ROOK)                                         ? 1
                     : t - p >= 2 && (b & 0x55AA55AA55AA55AAULL) && (b & ~0x55AA55AA55AA55AAULL) ? 2
                                                                                                 : 3;
    return MAT_ID_TABLE[t][p][h];
}

}  // namespace

namespace Detail {

// Read evaluation function parameters
template<typename T>
bool read_parameters(std::istream& stream, T& reference) {

    u32 header;
    header = read_little_endian<u32>(stream);
    if (!stream || header != T::get_hash_value())
        return false;
    return reference.read_parameters(stream);
}

// Write evaluation function parameters
template<typename T>
bool write_parameters(std::ostream& stream, const T& reference) {

    write_little_endian<u32>(stream, T::get_hash_value());
    return reference.write_parameters(stream);
}

}  // namespace Detail

void Network::load(const fs::path& rootDirectory, fs::path evalfilePath, EvalFile& evalFile) {
#if defined(DEFAULT_NNUE_DIRECTORY)
    std::vector<fs::path> dirs = {fs::path{}, rootDirectory,
                                  fs::path(stringify(DEFAULT_NNUE_DIRECTORY))};
#else
    std::vector<fs::path> dirs = {fs::path{}, rootDirectory};
#endif

    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (evalFile.current != evalfilePath && evalfilePath == evalFile.defaultName)
        load_internal(evalFile);

    for (const auto& directory : dirs)
    {
        if (evalFile.current != evalfilePath)
            load_external(directory, evalfilePath, evalFile);
    }
}

bool Network::save(const EvalFile& evalFile, const std::optional<fs::path>& filename) const {
    if (!evalFile.current.has_value())
    {
        sync_cout << "Failed to export a net. No network file is currently loaded. "
                     "Please load a network file first."
                  << sync_endl;
        return false;
    }

    if (!filename.has_value() && evalFile.current != evalFile.defaultName)
    {
        sync_cout << "Failed to export a net. A non-embedded net can only be "
                     "saved if the filename is specified"
                  << sync_endl;
        return false;
    }

    fs::path      actualFilename = filename.value_or(evalFile.defaultName);
    std::ofstream stream(actualFilename, std::ios_base::binary);

    bool saved = save(stream, evalFile.netDescription);

    sync_cout << (saved ? "Network saved successfully to " + actualFilename.string()
                        : "Failed to export a net")
              << sync_endl;

    return saved;
}

Value Network::evaluate(const Position&    pos,
                        AccumulatorStack&  accumulatorStack,
                        AccumulatorCaches& cache) const {

    constexpr u64 alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NNZInfo<L1> nnzInfo;

    const Color stm = pos.side_to_move();

    // Perspective 1 (STM):
    const uint16_t mat_id_us = get_material_id(pos, stm);
    const Square ourKsq = pos.square<KING>(stm);
    const Square ourKsqOriented = Square(int(ourKsq) ^ (56 * int(stm)));
    const bool ourCastling = pos.can_castle(stm == WHITE ? WHITE_CASTLING : BLACK_CASTLING);
    int ourKingBucket;
    if (ourCastling && ourKsqOriented == SQ_E1)
        ourKingBucket = 32;
    else
    {
        const int r = int(ourKsqOriented) / 8;
        const int f = int(ourKsqOriented) % 8;
        ourKingBucket = r * 4 + (f < 4 ? f : 7 - f);
    }

    const Square enemyKsq = pos.square<KING>(~stm);
    const Square enemyKsqOriented = Square(int(enemyKsq) ^ (56 * int(stm)));
    const bool enemyCastling = pos.can_castle(stm == WHITE ? BLACK_CASTLING : WHITE_CASTLING);
    int enemyKingBucket;
    if (enemyCastling && enemyKsqOriented == SQ_E8)
        enemyKingBucket = 64;
    else
        enemyKingBucket = int(enemyKsqOriented);

    const int c_us = make_router_index(mat_id_us, ourKingBucket, enemyKingBucket);
    const uint32_t packed_1   = router[c_us];
    const int e1 = packed_1 & 0x3F;
    const int      b1         = (packed_1 >> 16) & 0x7F;
    const int32_t* dyn_bias_1 = dynamicBiasTable[b1];

    // Perspective 2 (Non-STM):
    const uint16_t mat_id_them = get_material_id(pos, ~stm);
    const Square theirKsqOriented = Square(int(enemyKsq) ^ (56 * int(~stm)));
    int theirKingBucket;
    if (enemyCastling && theirKsqOriented == SQ_E1)
        theirKingBucket = 32;
    else
    {
        const int r = int(theirKsqOriented) / 8;
        const int f = int(theirKsqOriented) % 8;
        theirKingBucket = r * 4 + (f < 4 ? f : 7 - f);
    }

    const Square ourKsqOrientedThem = Square(int(ourKsq) ^ (56 * int(~stm)));
    int ourKingBucketThem;
    if (ourCastling && ourKsqOrientedThem == SQ_E8)
        ourKingBucketThem = 64;
    else
        ourKingBucketThem = int(ourKsqOrientedThem);

    const int c_them = make_router_index(mat_id_them, theirKingBucket, ourKingBucketThem);
    const uint32_t packed_2   = router[c_them];
    const int      e2         = (packed_2 >> 8) & 0x3F;
    const int      b2         = (packed_2 >> 24) & 0x7F;
    const int32_t* dyn_bias_2 = dynamicBiasTable[NumBiasExperts + b2];

    featureTransformer.transform(pos, accumulatorStack, cache, transformedFeatures, nnzInfo);
    const auto positional = network[e1].propagate(
      transformedFeatures, nnzInfo, network[LayerStacks + e2], dyn_bias_1, dyn_bias_2);
    return static_cast<Value>(positional / OutputScale);
}


void Network::verify(const std::function<void(std::string_view)>& f,
                     const EvalFile&                              evalFile,
                     fs::path                                     evalfilePath) const {
    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (evalFile.current != evalfilePath)
    {
        if (f)
        {
            std::string msg1 =
              "Network evaluation parameters compatible with the engine must be available.";
            std::string msg2 =
              "The network file " + evalfilePath.string() + " was not loaded successfully.";
            std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                               "including the directory name, to the network file.";
            std::string msg4 = "The default net can be downloaded from: "
                               "https://tests.stockfishchess.org/api/nn/"
                             + std::string(evalFile.defaultName);
            std::string msg5 = "The engine will be terminated now.";

            std::string msg = "ERROR: " + msg1 + '\n' + "ERROR: " + msg2 + '\n' + "ERROR: " + msg3
                            + '\n' + "ERROR: " + msg4 + '\n' + "ERROR: " + msg5 + '\n';

            f(msg);
        }

        exit(EXIT_FAILURE);
    }

    if (f)
    {
        usize size =
          sizeof(featureTransformer) + sizeof(network) + sizeof(router) + sizeof(dynamicBiasTable);
        f("NNUE evaluation using " + evalfilePath.string() + " ("
          + std::to_string(size / (1024 * 1024)) + "MiB, ("
          + std::to_string(featureTransformer.InputDimensions) + ", "
          + std::to_string(network[0].TransformedFeatureDimensions) + ", "
          + std::to_string(network[0].FC_0_OUTPUTS) + ", " + std::to_string(network[0].FC_1_OUTPUTS)
          + ", 1))");
    }
}


NnueEvalTrace Network::trace_evaluate(const Position&    pos,
                                      AccumulatorStack&  accumulatorStack,
                                      AccumulatorCaches& cache) const {

    constexpr u64 alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    const Color stm = pos.side_to_move();

    // Perspective 1 (STM):
    const uint16_t mat_id_us = get_material_id(pos, stm);
    const Square ourKsq = pos.square<KING>(stm);
    const Square ourKsqOriented = Square(int(ourKsq) ^ (56 * int(stm)));
    const bool ourCastling = pos.can_castle(stm == WHITE ? WHITE_CASTLING : BLACK_CASTLING);
    int ourKingBucket;
    if (ourCastling && ourKsqOriented == SQ_E1)
        ourKingBucket = 32;
    else
    {
        const int r = int(ourKsqOriented) / 8;
        const int f = int(ourKsqOriented) % 8;
        ourKingBucket = r * 4 + (f < 4 ? f : 7 - f);
    }

    const Square enemyKsq = pos.square<KING>(~stm);
    const Square enemyKsqOriented = Square(int(enemyKsq) ^ (56 * int(stm)));
    const bool enemyCastling = pos.can_castle(stm == WHITE ? BLACK_CASTLING : WHITE_CASTLING);
    int enemyKingBucket;
    if (enemyCastling && enemyKsqOriented == SQ_E8)
        enemyKingBucket = 64;
    else
        enemyKingBucket = int(enemyKsqOriented);

    const int c_us = make_router_index(mat_id_us, ourKingBucket, enemyKingBucket);
    const uint32_t packed_1 = router[c_us];
    const int e1 = packed_1 & 0x3F;
    const int      b1       = (packed_1 >> 16) & 0x7F;

    // Perspective 2 (Non-STM):
    const uint16_t mat_id_them = get_material_id(pos, ~stm);
    const Square theirKsqOriented = Square(int(enemyKsq) ^ (56 * int(~stm)));
    int theirKingBucket;
    if (enemyCastling && theirKsqOriented == SQ_E1)
        theirKingBucket = 32;
    else
    {
        const int r = int(theirKsqOriented) / 8;
        const int f = int(theirKsqOriented) % 8;
        theirKingBucket = r * 4 + (f < 4 ? f : 7 - f);
    }

    const Square ourKsqOrientedThem = Square(int(ourKsq) ^ (56 * int(~stm)));
    int ourKingBucketThem;
    if (ourCastling && ourKsqOrientedThem == SQ_E8)
        ourKingBucketThem = 64;
    else
        ourKingBucketThem = int(ourKsqOrientedThem);

    const int c_them = make_router_index(mat_id_them, theirKingBucket, ourKingBucketThem);
    const uint32_t packed_2 = router[c_them];
    const int      e2       = (packed_2 >> 8) & 0x3F;
    const int      b2       = (packed_2 >> 24) & 0x7F;

    NnueEvalTrace t{};
    t.weightExperts = {e1, e2};
    t.biasExperts   = {b1, b2};
    NNZInfo<L1> nnzInfo;
    featureTransformer.transform(pos, accumulatorStack, cache, transformedFeatures, nnzInfo);
    // Counterfactuals exchange contexts, not IDs between independent banks.
    const uint32_t contexts[4][2] = {
      {packed_1, packed_2}, {packed_1, packed_1}, {packed_2, packed_2}, {packed_2, packed_1}};
    for (int i = 0; i < 4; ++i)
    {
        const uint32_t us = contexts[i][0], them = contexts[i][1];
        t.positional[i] = static_cast<Value>(
          network[us & 0x3F].propagate(transformedFeatures, nnzInfo,
                                       network[LayerStacks + ((them >> 8) & 0x3F)],
                                       dynamicBiasTable[(us >> 16) & 0x7F],
                                       dynamicBiasTable[NumBiasExperts + ((them >> 24) & 0x7F)])
                             / OutputScale);
    }

    return t;
}


void Network::load_external(const fs::path& dir, const fs::path& evalfilePath, EvalFile& evalFile) {
    std::ifstream stream(dir / evalfilePath, std::ios::binary);
    auto          description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalfilePath;
        evalFile.netDescription = description.value();
    }
}


void Network::load_internal(EvalFile& evalFile) {
    // C++ way to prepare a buffer for a memory stream
    class MemoryBuffer: public std::basic_streambuf<char> {
       public:
        MemoryBuffer(char* p, usize n) {
            setg(p, p, p + n);
            setp(p, p + n);
        }
    };

#ifdef UNIVERSAL_BINARY_MACOS_X86_SLICE
    if (gEmbeddedNNUEData == nullptr)  // failed embedded load
        return;
#endif

    MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(gEmbeddedNNUEData)),
                        usize(gEmbeddedNNUESize));

    std::istream stream(&buffer);
    auto         description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalFile.defaultName;
        evalFile.netDescription = description.value();
    }
}


void Network::initialize() {
    initialized = true;
    std::memset(router, 0, sizeof(router));
    std::memset(dynamicBiasTable, 0, sizeof(dynamicBiasTable));
}


bool Network::save(std::ostream& stream, const std::string& netDescription) const {
    return write_parameters(stream, netDescription);
}


std::optional<std::string> Network::load(std::istream& stream) {
    initialize();
    std::string description;

    return read_parameters(stream, description) ? std::make_optional(description) : std::nullopt;
}


usize Network::get_content_hash() const {
    if (!initialized)
        return 0;

    usize h = 0;
    hash_combine(h, featureTransformer);
    for (auto&& layerstack : network)
        hash_combine(h, layerstack);
    for (usize i = 0; i < RouterTableSize; ++i)
        hash_combine(h, router[i]);
    for (usize i = 0; i < 2 * NumBiasExperts; ++i)
        for (usize j = 0; j < BiasDimension; ++j)
            hash_combine(h, dynamicBiasTable[i][j]);
    return h;
}

// Read network header
bool Network::read_header(std::istream& stream, u32* hashValue, std::string* desc) const {
    u32 version, size;

    version    = read_little_endian<u32>(stream);
    *hashValue = read_little_endian<u32>(stream);
    size       = read_little_endian<u32>(stream);
    if (!stream || version != Version)
        return false;

    constexpr u32 Chunk = 4096;
    char          buf[Chunk];

    desc->clear();
    for (u32 remaining = size; remaining > 0;)
    {
        const u32 want = std::min(remaining, Chunk);
        stream.read(buf, want);
        const u32 got = u32(stream.gcount());
        desc->append(buf, got);
        if (got != want)
            return false;
        remaining -= want;
    }
    return !stream.fail();
}


// Write network header
bool Network::write_header(std::ostream& stream, u32 hashValue, const std::string& desc) const {
    write_little_endian<u32>(stream, Version);
    write_little_endian<u32>(stream, hashValue);
    write_little_endian<u32>(stream, u32(desc.size()));
    stream.write(&desc[0], desc.size());
    return !stream.fail();
}


bool Network::read_parameters(std::istream& stream, std::string& netDescription) {
    u32 hashValue;
    if (!read_header(stream, &hashValue, &netDescription))
        return false;
    if (hashValue != Network::hash)
        return false;
    if (!Detail::read_parameters(stream, featureTransformer))
        return false;
    for (usize i = 0; i < 2 * LayerStacks; ++i)
    {
        if (!Detail::read_parameters(stream, network[i]))
            return false;
    }
    read_little_endian<uint32_t>(stream, router, RouterTableSize);
    read_little_endian<int32_t>(stream, reinterpret_cast<int32_t*>(dynamicBiasTable),
                                2 * NumBiasExperts * BiasDimension);
    return stream && stream.peek() == std::ios::traits_type::eof();
}


bool Network::write_parameters(std::ostream& stream, const std::string& netDescription) const {
    if (!write_header(stream, Network::hash, netDescription))
        return false;
    if (!Detail::write_parameters(stream, featureTransformer))
        return false;
    for (usize i = 0; i < 2 * LayerStacks; ++i)
    {
        if (!Detail::write_parameters(stream, network[i]))
            return false;
    }
    write_little_endian<uint32_t>(stream, router, RouterTableSize);
    write_little_endian<int32_t>(stream, reinterpret_cast<const int32_t*>(dynamicBiasTable),
                                 2 * NumBiasExperts * BiasDimension);
    return bool(stream);
}

}  // namespace Stockfish::Eval::NNUE
