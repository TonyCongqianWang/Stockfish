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

constexpr IndexType LayerStacks        = 64;
constexpr IndexType RouterTableSize    = 849420;
constexpr IndexType NumFC0Experts      = 64;
constexpr IndexType NumMaterialStates  = 396;
constexpr IndexType FC0RouterTableSize = NumMaterialStates * NumMaterialStates;
constexpr u32       RouterHash         = 0x4D6F4537u;
constexpr IndexType NumBiasExperts     = 64;
constexpr IndexType BiasDimension      = 64;

static constexpr u32 get_router_hash() {
    return RouterHash ^ ((NumBiasExperts << 16) | (LayerStacks << 8) | NumFC0Experts);
}

struct InputLayer {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS, FC_0_OUTPUTS * 2>
                                                              fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 2> ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 2>    ac_0;

    static constexpr u32 get_hash_value() {
        u32 hash = 0xEC42E90Du ^ (TransformedFeatureDimensions * 2);
        return decltype(ac_0)::get_hash_value(decltype(fc_0)::get_hash_value(hash));
    }

    bool  read_parameters(std::istream& stream) { return fc_0.read_parameters(stream); }
    bool  write_parameters(std::ostream& stream) const { return fc_0.write_parameters(stream); }
    usize get_content_hash() const { return fc_0.get_content_hash(); }
};

struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;
    static constexpr int       FC_1_OUTPUTS                 = L3 / 2;

    Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS, FC_1_OUTPUTS * 2> fc_1;
    Layers::SqrClippedReLU<FC_1_OUTPUTS * 2, WeightScaleBits + 1>             ac_sqr_1;
    Layers::ClippedReLU<FC_1_OUTPUTS * 2, WeightScaleBits + 1>                ac_1;
    Layers::AffineTransform<FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 4, 1>           fc_2;

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        u32 hashValue = InputLayer::get_hash_value();
        hashValue     = decltype(fc_1)::get_hash_value(hashValue);
        hashValue     = decltype(ac_1)::get_hash_value(hashValue);
        hashValue     = decltype(fc_2)::get_hash_value(hashValue);

        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        return fc_1.read_parameters(stream) && ac_1.read_parameters(stream)
            && fc_2.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        return fc_1.write_parameters(stream) && ac_1.write_parameters(stream)
            && fc_2.write_parameters(stream);
    }

    i32 propagate(const TransformedFeatureType* transformedFeatures,
                  const NNZInfo<L1>&            nnzInfo,
                  const InputLayer&             firstLayer,
                  const NetworkArchitecture&    otherTail,
                  const int32_t*                dynBias,
                  const int32_t*                otherBias) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(InputLayer::fc_0)::OutputBuffer first;
            alignas(CacheLineSize) i32 firstSqr[FC_0_OUTPUTS], firstLin[FC_0_OUTPUTS];
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer second[2];
            alignas(CacheLineSize) i32 secondSqr[FC_1_OUTPUTS * 2], secondLin[FC_1_OUTPUTS * 2];
            alignas(CacheLineSize) u8 combined[FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 4];
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer output[2];
        };
        Buffer buffer;
        firstLayer.fc_0.propagate(transformedFeatures, buffer.first, nnzInfo);
        constexpr IndexType Half  = FC_0_OUTPUTS / 2;
        const i64           skips = i64(buffer.first[Half - 2]) - buffer.first[Half - 1]
                        + buffer.first[FC_0_OUTPUTS - 2] - buffer.first[FC_0_OUTPUTS - 1];
        const int32_t* biases[2] = {dynBias, otherBias};
        const auto add_biases = [&](const auto& projections, const i32* ownBias, const i32* oppBias,
                                    auto& sqr, auto& lin, IndexType width, IndexType offset) {
            const i32* expertBiases[2] = {ownBias, oppBias};
            for (int p = 0; p < 2; ++p)
                for (IndexType i = 0; i < width; ++i)
                {
                    sqr[p * width + i] =
                      i32(i64(projections[p][i]) + expertBiases[p][i] + biases[p][offset + i]);
                    lin[p * width + i] = i32(i64(projections[p][i]) + expertBiases[p][width + i]
                                             + biases[p][offset + width + i]);
                }
        };
        const i32* staticBias = firstLayer.fc_0.get_biases();
        for (int p = 0; p < 2; ++p)
            for (IndexType i = 0; i < Half; ++i)
            {
                const IndexType j  = p * Half + i;
                buffer.firstSqr[j] = i32(i64(buffer.first[j]) + staticBias[j] + biases[p][i]);
                buffer.firstLin[j] =
                  i32(i64(buffer.first[j]) + staticBias[FC_0_OUTPUTS + j] + biases[p][Half + i]);
            }
#if defined(USE_PAIR_ACTIVATIONS)
        firstLayer.ac_sqr_0.propagate_pair(buffer.firstSqr, buffer.firstLin, buffer.combined,
                                           buffer.combined + FC_0_OUTPUTS);
#else
        firstLayer.ac_sqr_0.propagate(buffer.firstSqr, buffer.combined);
        firstLayer.ac_0.propagate(buffer.firstLin, buffer.combined + FC_0_OUTPUTS);
#endif
        fc_1.propagate_pair(buffer.combined, buffer.second[0], otherTail.fc_1, buffer.second[1]);
        add_biases(buffer.second, fc_1.get_biases(), otherTail.fc_1.get_biases(), buffer.secondSqr,
                   buffer.secondLin, FC_1_OUTPUTS, FC_0_OUTPUTS);
        constexpr IndexType Offset = FC_0_OUTPUTS * 2;
#if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_1.propagate_pair(buffer.secondSqr, buffer.secondLin, buffer.combined + Offset,
                                buffer.combined + Offset + FC_1_OUTPUTS * 2);
#else
        ac_sqr_1.propagate(buffer.secondSqr, buffer.combined + Offset);
        ac_1.propagate(buffer.secondLin, buffer.combined + Offset + FC_1_OUTPUTS * 2);
#endif
        fc_2.propagate(buffer.combined, buffer.output[0]);
        otherTail.fc_2.propagate(buffer.combined, buffer.output[1]);
        // Both readouts predict STM scores. Convert their scale 128 to the
        // skips' scale 256, then average complete predictions exactly once.
        const i64     fwdOut = (2 * (i64(buffer.output[0][0]) + buffer.output[1][0]) + skips) / 2;
        constexpr i64 multiplier  = 600 * OutputScale;
        constexpr i64 denominator = static_cast<i64>(HiddenOneVal) * (1U << WeightScaleBits) * 4;

        i32 outputValue = static_cast<i32>((static_cast<i64>(fwdOut) * multiplier) / denominator);
        return outputValue;
    }

    usize get_content_hash() const {
        usize h = 0;
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
