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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>
#include <filesystem>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../incbin/incbin.h"

#include "../bitboard.h"
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
        if (evalFile.current != evalfilePath)
            load_external(directory, evalfilePath, evalFile);
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

namespace {

constexpr IndexType MatchedControlStacks = 64;

int moe11_context(const Position& pos) {
    const auto side = [&](Color c) {
        return Moe11::side(pos.count<ALL_PIECES>(c), pos.count<PAWN>(c), pos.count<ROOK>(c),
                           pos.count<QUEEN>(c), pos.pieces(c, BISHOP), pos.pieces(c, PAWN));
    };
    const Color stm = pos.side_to_move();
    return Moe11::context(side(stm), side(~stm), int(pos.square<KING>(stm)),
                          int(pos.square<KING>(~stm)), stm == BLACK);
}

}  // namespace

int Network::layer_stack(const Position& pos) const {
    if (learnedRouting)
        return routeLut[moe11_context(pos)];

    const Color stm    = pos.side_to_move();
    const int   band   = (pos.count<ALL_PIECES>() - 1) / 4;
    const int   queens = 2 * bool(pos.pieces(stm, QUEEN)) + bool(pos.pieces(~stm, QUEEN));

    // Officers are the pieces other than pawns and kings
    const auto officers = [&](Color c) {
        return popcount(pos.pieces(c) & ~pos.pieces(PAWN, KING));
    };
    const bool many = std::max(officers(WHITE), officers(BLACK)) >= 2;

    if (rawLayerStacks == MatchedControlStacks)
        return layerStackIndex[(band * 4 + queens) * 2 + many];

    // Pawn layout: 0 if pawns stand on both board halves, 1 if on one half only, 2 if there are no pawns
    constexpr Bitboard QueenSide = FileABB | FileBBB | FileCBB | FileDBB;
    const Bitboard     pawns     = pos.pieces(PAWN);
    const int          layout    = !pawns ? 2 : (pawns & QueenSide) && (pawns & ~QueenSide) ? 0 : 1;

    return layerStackIndex[((band * 4 + queens) * 3 + layout) * 2 + many];
}

Value Network::evaluate(const Position&    pos,
                        AccumulatorStack&  accumulatorStack,
                        AccumulatorCaches& cache) const {

    constexpr u64 alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NNZInfo<L1> nnzInfo;

    const int bucket = layer_stack(pos);

    featureTransformer.transform(pos, accumulatorStack, cache, transformedFeatures, nnzInfo);
    const auto positional = network[bucket].propagate(transformedFeatures, nnzInfo);
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
        usize size = sizeof(featureTransformer) + sizeof(NetworkArchitecture) * storedLayerStacks
                   + (learnedRouting ? routeLut.size() : 0);
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

    const int context = learnedRouting ? moe11_context(pos) : -1;
    const int bucket  = learnedRouting ? routeLut[context] : layer_stack(pos);

    NnueEvalTrace t{};
    t.correctBucket = bucket;
    t.layerStacks   = storedLayerStacks;
    t.contextId     = context;

    NNZInfo<L1> nnzInfo;
    featureTransformer.transform(pos, accumulatorStack, cache, transformedFeatures, nnzInfo);

    const IndexType first = learnedRouting ? bucket : 0;
    const IndexType end   = learnedRouting ? bucket + 1 : storedLayerStacks;
    for (IndexType b = first; b < end; ++b)
    {
        const auto positional = network[b].propagate(transformedFeatures, nnzInfo);
        t.positional[b]       = static_cast<Value>(positional / OutputScale);
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


void Network::initialize() { initialized = true; }


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
    if (learnedRouting)
    {
        hash_combine(h, Moe11::SchemaHash);
        hash_combine(h, get_raw_data_hash(routeLut));
    }
    else
        for (IndexType s = 0; s < rawLayerStacks; ++s)
            hash_combine(h, layerStackFallback[s]);
    for (usize i = 0; i < storedLayerStacks; ++i)
        hash_combine(h, network[i]);
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
    const u32 tableHash = read_little_endian<u32>(stream);
    if (hashValue != (architectureHash ^ tableHash))
        return false;
    const auto raw    = read_little_endian<u32>(stream);
    const auto stored = read_little_endian<u32>(stream);
    if (tableHash == Moe11::SchemaHash)
    {
        if (raw != Moe11::ContextCount || stored != Moe11::ExpertCount)
            return false;
        stream.read(reinterpret_cast<char*>(routeLut.data()), routeLut.size());
        if (!stream || std::any_of(routeLut.begin(), routeLut.end(), [](u8 id) {
                return id >= Moe11::ExpertCount;
            }))
            return false;
        learnedRouting = true;
    }
    else if (tableHash == LayerStackTableHash || tableHash == MatchedControlHash)
    {
        const IndexType expected =
          tableHash == MatchedControlHash ? MatchedControlStacks : RawLayerStacks;
        if (raw != expected || !stored || stored > raw)
            return false;
        IndexType count = 0;
        for (IndexType s = 0; s < raw; ++s)
        {
            const u16 f = read_little_endian<u16>(stream);
            if (f >= raw)
                return false;
            layerStackFallback[s] = f;
            if (f == s)
                layerStackIndex[s] = u16(count++);
        }
        if (count != stored)
            return false;
        for (IndexType s = 0; s < raw; ++s)
        {
            const u16 f = layerStackFallback[s];
            if (layerStackFallback[f] != f)
                return false;
            layerStackIndex[s] = layerStackIndex[f];
        }
        learnedRouting = false;
        rawLayerStacks = raw;
    }
    else
        return false;
    storedLayerStacks = stored;
    if (!Detail::read_parameters(stream, featureTransformer))
        return false;
    for (usize i = 0; i < storedLayerStacks; ++i)
    {
        if (!Detail::read_parameters(stream, network[i]))
            return false;
    }
    return stream && stream.peek() == std::ios::traits_type::eof();
}


bool Network::write_parameters(std::ostream& stream, const std::string& netDescription) const {
    const u32 tableHash = learnedRouting                         ? Moe11::SchemaHash
                        : rawLayerStacks == MatchedControlStacks ? MatchedControlHash
                                                                 : LayerStackTableHash;
    if (!write_header(stream, architectureHash ^ tableHash, netDescription))
        return false;
    write_little_endian<u32>(stream, tableHash);
    write_little_endian<u32>(stream, learnedRouting ? Moe11::ContextCount : rawLayerStacks);
    write_little_endian<u32>(stream, storedLayerStacks);
    if (learnedRouting)
        stream.write(reinterpret_cast<const char*>(routeLut.data()), routeLut.size());
    else
        for (IndexType s = 0; s < rawLayerStacks; ++s)
            write_little_endian<u16>(stream, layerStackFallback[s]);
    if (!Detail::write_parameters(stream, featureTransformer))
        return false;
    for (usize i = 0; i < storedLayerStacks; ++i)
    {
        if (!Detail::write_parameters(stream, network[i]))
            return false;
    }
    return bool(stream);
}

}  // namespace Stockfish::Eval::NNUE
