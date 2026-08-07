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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <iterator>

#include "../position.h"
#include "nnz_helper.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "simd.h"

namespace Stockfish::Eval::NNUE {

// Returns the inverse of a permutation
template<usize Len>
constexpr std::array<usize, Len> invert_permutation(const std::array<usize, Len>& order) {
    std::array<usize, Len> inverse{};
    for (usize i = 0; i < order.size(); i++)
        inverse[order[i]] = i;
    return inverse;
}

// Divide a byte region of size TotalSize to chunks of size
// BlockSize, and permute the blocks by a given order
template<usize BlockSize, typename T, usize N, usize OrderSize>
void permute(std::array<T, N>& data, const std::array<usize, OrderSize>& order) {
    constexpr usize TotalSize = N * sizeof(T);

    static_assert(TotalSize % (BlockSize * OrderSize) == 0,
                  "ChunkSize * OrderSize must perfectly divide TotalSize");

    constexpr usize ProcessChunkSize = BlockSize * OrderSize;

    std::array<std::byte, ProcessChunkSize> buffer{};

    std::byte* const bytes = reinterpret_cast<std::byte*>(data.data());

    for (usize i = 0; i < TotalSize; i += ProcessChunkSize)
    {
        std::byte* const values = &bytes[i];

        for (usize j = 0; j < OrderSize; j++)
        {
            auto* const buffer_chunk = &buffer[j * BlockSize];
            auto* const value_chunk  = &values[order[j] * BlockSize];

            std::copy(value_chunk, value_chunk + BlockSize, buffer_chunk);
        }

        std::copy(std::begin(buffer), std::end(buffer), values);
    }
}

// Input feature converter
class FeatureTransformer {
    // Number of output dimensions for one side
    static constexpr IndexType HalfDimensions = L1;

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType ThreatInputDimensions = ThreatFeatureSet::Dimensions;
    static constexpr IndexType PairInputDimensions   = PairFeatureSet::Dimensions;
    static constexpr IndexType InputDimensions =
      PSQFeatureSet::Dimensions + ThreatInputDimensions + PairInputDimensions;
    static constexpr IndexType OutputDimensions = HalfDimensions;

    // Size of forward propagation buffer
    static constexpr usize BufferSize = OutputDimensions * sizeof(OutputType);

    // Store the order by which 128-bit blocks of a 1024-bit data must
    // be permuted so that calling packus on adjacent vectors of 16-bit
    // integers loaded from the data results in the pre-permutation order
    static constexpr auto PackusEpi16Order = []() -> std::array<usize, 8> {
#if defined(USE_AVX512)
        // _mm512_packus_epi16 after permutation:
        // |   0   |   2   |   4   |   6   | // Vector 0
        // |   1   |   3   |   5   |   7   | // Vector 1
        // | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | // Packed Result
        return {0, 2, 4, 6, 1, 3, 5, 7};
#elif defined(USE_AVX2) || defined(USE_LASX)
        // _mm256_packus_epi16 after permutation:
        // |   0   |   2   |  |   4   |   6   | // Vector 0, 2
        // |   1   |   3   |  |   5   |   7   | // Vector 1, 3
        // | 0 | 1 | 2 | 3 |  | 4 | 5 | 6 | 7 | // Packed Result
        return {0, 2, 1, 3, 4, 6, 5, 7};
#else
        return {0, 1, 2, 3, 4, 5, 6, 7};
#endif
    }();

    static constexpr auto InversePackusEpi16Order = invert_permutation(PackusEpi16Order);

    static constexpr u32 combine_hash(std::initializer_list<u32> hashes) {
        u32 hash = 0;
        for (const auto component_hash : hashes)
        {
            hash = (hash << 1) | (hash >> 31);
            hash ^= component_hash;
        }
        return hash;
    }

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        return combine_hash(
                 {ThreatFeatureSet::HashValue, PairFeatureSet::HashValue, PSQFeatureSet::HashValue})
             ^ (OutputDimensions * 2);
    }

    void permute_weights() {
        permute<16>(biases, PackusEpi16Order);
        permute<16>(weights, PackusEpi16Order);

        permute<8>(threatAndPpWeights, PackusEpi16Order);
    }

    void unpermute_weights() {
        permute<16>(biases, InversePackusEpi16Order);
        permute<16>(weights, InversePackusEpi16Order);
        permute<8>(threatAndPpWeights, InversePackusEpi16Order);
    }

    auto threatWeights() { return threatAndPpWeights.data(); }
    auto threatWeights() const { return threatAndPpWeights.data(); }
    auto ppWeights() { return &threatAndPpWeights[ThreatFeatureSet::Dimensions * HalfDimensions]; }
    auto ppWeights() const {
        return &threatAndPpWeights[ThreatFeatureSet::Dimensions * HalfDimensions];
    }

    auto threatPsqtWeights() { return threatAndPpPsqtWeights.data(); }
    auto threatPsqtWeights() const { return threatAndPpPsqtWeights.data(); }
    auto ppPsqtWeights() {
        return &threatAndPpPsqtWeights[ThreatFeatureSet::Dimensions * PSQTBuckets];
    }
    auto ppPsqtWeights() const {
        return &threatAndPpPsqtWeights[ThreatFeatureSet::Dimensions * PSQTBuckets];
    }


    // Read network parameters
    bool read_parameters(std::istream& stream) {
        read_leb_128(stream, biases);

        read_little_endian<ThreatWeightType>(stream, threatWeights(),
                                             ThreatInputDimensions * HalfDimensions);
        read_leb_128(stream, threatPsqtWeights(), ThreatFeatureSet::Dimensions * PSQTBuckets);
        read_little_endian<ThreatWeightType>(stream, ppWeights(),
                                             PairInputDimensions * HalfDimensions);
        read_leb_128(stream, ppPsqtWeights(), PairFeatureSet::Dimensions * PSQTBuckets);

        read_leb_128(stream, weights);
        read_leb_128(stream, psqtWeights);

        permute_weights();

        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        std::unique_ptr<FeatureTransformer> copy = std::make_unique<FeatureTransformer>(*this);

        copy->unpermute_weights();

        write_leb_128<BiasType>(stream, copy->biases);


        write_little_endian<ThreatWeightType>(stream, copy->threatWeights(),
                                              ThreatInputDimensions * HalfDimensions);
        write_leb_128<PSQTWeightType>(stream, copy->threatPsqtWeights(),
                                      ThreatFeatureSet::Dimensions * PSQTBuckets);
        write_little_endian<ThreatWeightType>(stream, copy->ppWeights(),
                                              PairInputDimensions * HalfDimensions);
        write_leb_128<PSQTWeightType>(stream, copy->ppPsqtWeights(),
                                      PairFeatureSet::Dimensions * PSQTBuckets);

        write_leb_128<WeightType>(stream, copy->weights);
        write_leb_128<PSQTWeightType>(stream, copy->psqtWeights);

        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;

        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_raw_data_hash(psqtWeights));

        hash_combine(h, get_raw_data_hash(threatAndPpWeights));
        hash_combine(h, get_raw_data_hash(threatAndPpPsqtWeights));

        hash_combine(h, get_hash_value());

        return h;
    }

    // Convert input features
    i32 transform(const Position&                             pos,
                  AccumulatorStack&                           accumulatorStack,
                  AccumulatorCaches&                          cache,
                  OutputType*                                 output,
                  int                                         bucket,
                  [[maybe_unused]] NNZInfo<OutputDimensions>& nnzInfo) const {

        using namespace SIMD;
        accumulatorStack.evaluate(pos, *this, cache);
        const auto& accumulatorState = accumulatorStack.latest();

        const Color perspectives[2]  = {pos.side_to_move(), ~pos.side_to_move()};
        const auto& psqtAccumulation = accumulatorState.psqtAccumulation;
        const auto  psqt =
          (psqtAccumulation[perspectives[0]][bucket] - psqtAccumulation[perspectives[1]][bucket])
          / 2;

        const auto& accumulation = accumulatorState.accumulation;
        const auto& us_acc   = accumulation[perspectives[0]];
        const auto& them_acc = accumulation[perspectives[1]];
        constexpr IndexType Q = HalfDimensions / 4;

        // Quarter 0: w0 * w1
        for (IndexType j = 0; j < Q; ++j)
        {
            BiasType sum0 = std::clamp<BiasType>(us_acc[0 * Q + j], 0, FtMaxVal);
            BiasType sum1 = std::clamp<BiasType>(us_acc[1 * Q + j], 0, FtMaxVal);
            output[0 * Q + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
        }
        // Quarter 1: b0 * b1
        for (IndexType j = 0; j < Q; ++j)
        {
            BiasType sum0 = std::clamp<BiasType>(them_acc[0 * Q + j], 0, FtMaxVal);
            BiasType sum1 = std::clamp<BiasType>(them_acc[1 * Q + j], 0, FtMaxVal);
            output[1 * Q + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
        }
        // Quarter 2: w2 * b3
        for (IndexType j = 0; j < Q; ++j)
        {
            BiasType sum0 = std::clamp<BiasType>(us_acc[2 * Q + j], 0, FtMaxVal);
            BiasType sum1 = std::clamp<BiasType>(them_acc[3 * Q + j], 0, FtMaxVal);
            output[2 * Q + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
        }
        // Quarter 3: b2 * w3
        for (IndexType j = 0; j < Q; ++j)
        {
            BiasType sum0 = std::clamp<BiasType>(them_acc[2 * Q + j], 0, FtMaxVal);
            BiasType sum1 = std::clamp<BiasType>(us_acc[3 * Q + j], 0, FtMaxVal);
            output[3 * Q + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 512);
        }

        std::memset(nnzInfo.bitset, 0xFF, sizeof(nnzInfo.bitset));

        return psqt;
    }  // end of function transform()

    alignas(CacheLineSize) std::array<BiasType, HalfDimensions> biases;
    alignas(
      CacheLineSize) std::array<WeightType, HalfDimensions * PSQFeatureSet::Dimensions> weights;

    // Threats and pawn-pair features are concatenated into one array to allow for a single index to address either.
    // The first pawn-pair feature is at index ThreatFeatureSet::Dimensions.
    static_assert(PairFeatureSet::IndexBase == ThreatFeatureSet::Dimensions);

    alignas(CacheLineSize) std::array<ThreatWeightType,
                                      (ThreatFeatureSet::Dimensions + PairFeatureSet::Dimensions)
                                        * HalfDimensions> threatAndPpWeights;
    alignas(CacheLineSize)
      std::array<PSQTWeightType, PSQTBuckets * PSQFeatureSet::Dimensions> psqtWeights;
    // As above
    alignas(CacheLineSize) std::array<PSQTWeightType,
                                      (ThreatFeatureSet::Dimensions + PairFeatureSet::Dimensions)
                                        * PSQTBuckets> threatAndPpPsqtWeights;
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::FeatureTransformer> {
    Stockfish::usize
    operator()(const Stockfish::Eval::NNUE::FeatureTransformer& ft) const noexcept {
        return ft.get_content_hash();
    }
};

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
