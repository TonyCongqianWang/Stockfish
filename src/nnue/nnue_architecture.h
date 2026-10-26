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

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "features/k32h3.h"
#include "features/full_threats.h"
#include "features/pp_3wide.h"
#include "layers/affine_transform.h"
#include "layers/affine_transform_sparse_input.h"
#include "layers/clipped_relu.h"
#include "layers/sqr_clipped_relu.h"
#include "nnue_common.h"
#include "nnz_helper.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using ThreatFeatureSet = Features::FullThreats;
using PairFeatureSet   = Features::PP_3Wide;
using PSQFeatureSet    = Features::K32H3;

// Number of input feature dimensions after conversion
constexpr IndexType L1 = 1024;
constexpr int       L2 = 32;
constexpr int       L3 = 32;

constexpr IndexType LayerStacks     = 64;
constexpr IndexType RouterTableSize = 849420;
constexpr u32       RouterHash      = 0x4D6F4535u;
constexpr IndexType NumBiasExperts  = 256;
constexpr IndexType BiasDimension   = 64;

static constexpr u32 get_router_hash() {
    return RouterHash ^ ((NumBiasExperts << 16) | LayerStacks);
}

struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2 / 2;
    static constexpr int       FC_1_OUTPUTS                 = L3 / 2;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS> fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS * 2, WeightScaleBits + 1>                  ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS * 2, WeightScaleBits + 1>                     ac_0;
    Layers::AffineTransform<FC_0_OUTPUTS * 4, FC_1_OUTPUTS>                        fc_1;
    Layers::SqrClippedReLU<FC_1_OUTPUTS * 2, WeightScaleBits>                      ac_sqr_1;
    Layers::ClippedReLU<FC_1_OUTPUTS * 2, WeightScaleBits>                         ac_1;
    Layers::AffineTransform<FC_0_OUTPUTS * 4 + FC_1_OUTPUTS * 4, 1>                fc_2;

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        // input slice hash
        u32 hashValue = 0xEC42E90Du;
        hashValue ^= TransformedFeatureDimensions * 2;

        hashValue = decltype(fc_0)::get_hash_value(hashValue);
        // TODO: consider including hash value of ac_sqr_0 in the overall hash value.
        // For now omitted on purpose because hash value is not written by trainer (yet)
        hashValue = decltype(ac_0)::get_hash_value(hashValue);
        hashValue = decltype(fc_1)::get_hash_value(hashValue);
        hashValue = decltype(ac_1)::get_hash_value(hashValue);
        hashValue = decltype(fc_2)::get_hash_value(hashValue);

        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        return fc_0.read_parameters(stream) && ac_0.read_parameters(stream)
            && fc_1.read_parameters(stream) && ac_1.read_parameters(stream)
            && fc_2.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        return fc_0.write_parameters(stream) && ac_0.write_parameters(stream)
            && fc_1.write_parameters(stream) && ac_1.write_parameters(stream)
            && fc_2.write_parameters(stream);
    }

    i32 propagate(const TransformedFeatureType* transformedFeatures,
                  const NNZInfo<L1>&            nnzInfo,
                  const NetworkArchitecture&    other,
                  const int32_t*                dynBias,
                  const int32_t*                otherBias) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) TransformedFeatureType reversedFT[L1];
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer first[2];
            alignas(CacheLineSize) i32 firstSqr[FC_0_OUTPUTS * 2], firstLin[FC_0_OUTPUTS * 2];
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer second[2];
            alignas(CacheLineSize) i32 secondSqr[FC_1_OUTPUTS * 2], secondLin[FC_1_OUTPUTS * 2];
            alignas(CacheLineSize) u8 combined[(FC_0_OUTPUTS + FC_1_OUTPUTS) * 4];
            alignas(CacheLineSize) u8 reversed[(FC_0_OUTPUTS + FC_1_OUTPUTS) * 4];
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer output[2];
        };
        Buffer buffer;
        std::rotate_copy(transformedFeatures, transformedFeatures + L1 / 2,
                         transformedFeatures + L1, buffer.reversedFT);
        NNZInfo<L1> reversedNNZ;
#if defined(USE_AVX512)
        reversedNNZ.count = nnzInfo.count;
        for (unsigned i = 0; i < nnzInfo.count; ++i)
            reversedNNZ.nnz[i] = nnzInfo.nnz[i] ^ (L1 / 8);  // Four-byte chunks
#elif defined(USE_RVV)
        reversedNNZ.count = nnzInfo.count;
        for (usize i = 0; i < nnzInfo.count; ++i)
            reversedNNZ.nnz[i] = nnzInfo.nnz[i] ^ (L1 / 2);
#else
        constexpr usize HalfNNZ = sizeof(nnzInfo.bitset) / 2;
        std::memcpy(reversedNNZ.bitset, nnzInfo.bitset + HalfNNZ, HalfNNZ);
        std::memcpy(reversedNNZ.bitset + HalfNNZ, nnzInfo.bitset, HalfNNZ);
#endif
        fc_0.propagate(transformedFeatures, buffer.first[0], nnzInfo);
        other.fc_0.propagate(buffer.reversedFT, buffer.first[1], reversedNNZ);
        const i32 skip =
          i32(((i64(buffer.first[0][FC_0_OUTPUTS - 2]) - buffer.first[0][FC_0_OUTPUTS - 1])
               - (i64(buffer.first[1][FC_0_OUTPUTS - 2]) - buffer.first[1][FC_0_OUTPUTS - 1]))
              / 2);
        const int32_t* biases[2] = {dynBias, otherBias};
        for (int p = 0; p < 2; ++p)
            for (IndexType i = 0; i < FC_0_OUTPUTS; ++i)
            {
                buffer.firstSqr[p * FC_0_OUTPUTS + i] = i32(i64(buffer.first[p][i]) + biases[p][i]);
                buffer.firstLin[p * FC_0_OUTPUTS + i] =
                  i32(i64(buffer.first[p][i]) + biases[p][FC_0_OUTPUTS + i]);
            }
#if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_0.propagate_pair(buffer.firstSqr, buffer.firstLin, buffer.combined,
                                buffer.combined + FC_0_OUTPUTS * 2);
#else
        ac_sqr_0.propagate(buffer.firstSqr, buffer.combined);
        ac_0.propagate(buffer.firstLin, buffer.combined + FC_0_OUTPUTS * 2);
#endif
        // Logical activations are [A_sq, B_sq, A_lin, B_lin]. Account for
        // SIMD lane packing when exchanging the two 16-neuron perspectives.
        static_assert(FC_0_OUTPUTS == 16 && FC_1_OUTPUTS == 16);
#if defined(USE_PAIR_ACTIVATIONS) && defined(USE_AVX512)
        constexpr IndexType PerspectiveMask = 4;
#elif defined(USE_PAIR_ACTIVATIONS) || defined(USE_LASX)
        constexpr IndexType PerspectiveMask = 8;
#else
        constexpr IndexType PerspectiveMask = 16;
#endif
        for (IndexType i = 0; i < FC_0_OUTPUTS * 4; ++i)
            buffer.reversed[i] = buffer.combined[i ^ PerspectiveMask];
        fc_1.propagate(buffer.combined, buffer.second[0]);
        other.fc_1.propagate(buffer.reversed, buffer.second[1]);
        for (int p = 0; p < 2; ++p)
            for (IndexType i = 0; i < FC_1_OUTPUTS; ++i)
            {
                buffer.secondSqr[p * FC_1_OUTPUTS + i] =
                  i32(i64(buffer.second[p][i]) + biases[p][FC_0_OUTPUTS * 2 + i]);
                buffer.secondLin[p * FC_1_OUTPUTS + i] =
                  i32(i64(buffer.second[p][i]) + biases[p][FC_0_OUTPUTS * 2 + FC_1_OUTPUTS + i]);
            }
        constexpr IndexType Offset = FC_0_OUTPUTS * 4;
#if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_1.propagate_pair(buffer.secondSqr, buffer.secondLin, buffer.combined + Offset,
                                buffer.combined + Offset + FC_1_OUTPUTS * 2);
#else
        ac_sqr_1.propagate(buffer.secondSqr, buffer.combined + Offset);
        ac_1.propagate(buffer.secondLin, buffer.combined + Offset + FC_1_OUTPUTS * 2);
#endif
        for (IndexType i = 0; i < FC_1_OUTPUTS * 4; ++i)
            buffer.reversed[Offset + i] = buffer.combined[Offset + (i ^ PerspectiveMask)];
        fc_2.propagate(buffer.combined, buffer.output[0]);
        other.fc_2.propagate(buffer.reversed, buffer.output[1]);
        // Convert the opponent readout and skip to STM coordinates.
        i32 fwdOut = i32((i64(buffer.output[0][0]) - buffer.output[1][0]) / 2) + skip;

        // fwdOut is such that 1.0 is equal to HiddenOneVal*(1<<WeightScaleBits)*2 in
        // quantized form, but we want 1.0 to be equal to 600*OutputScale
        // to make overflow impossible we cast to int64_t
        constexpr i64 multiplier = 600 * OutputScale;
        constexpr i64 denominator =
          static_cast<i64>(HiddenOneVal) * static_cast<i64>(1U << WeightScaleBits) * 2;

        i32 outputValue = static_cast<i32>((static_cast<i64>(fwdOut) * multiplier) / denominator);
        return outputValue;
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, fc_0.get_content_hash());
        hash_combine(h, ac_sqr_0.get_content_hash());
        hash_combine(h, ac_0.get_content_hash());
        hash_combine(h, fc_1.get_content_hash());
        // hash_combine(h, ac_sqr_1.get_content_hash()); TODO
        hash_combine(h, ac_1.get_content_hash());
        hash_combine(h, fc_2.get_content_hash());
        hash_combine(h, get_hash_value());
        return h;
    }
};

}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::NetworkArchitecture> {
    Stockfish::usize
    operator()(const Stockfish::Eval::NNUE::NetworkArchitecture& arch) const noexcept {
        return arch.get_content_hash();
    }
};

#endif  // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
