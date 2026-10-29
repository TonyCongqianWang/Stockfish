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

#include <cstdint>
#include <algorithm>
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
#include "perspective_buckets.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using ThreatFeatureSet = Features::FullThreats;
using PairFeatureSet   = Features::PP_3Wide;
using PSQFeatureSet    = Features::K32H3;

// Number of input feature dimensions after conversion
constexpr IndexType L1 = 1024;
constexpr int       L2 = 32;
constexpr int       L3 = 32;

constexpr IndexType LayerStacks = RoleExperts * 2;

struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2 / 2;
    static constexpr int       FC_1_OUTPUTS                 = L3 / 2;

    Layers::AffineTransformSparseInput<L1, FC_0_OUTPUTS> fc_0;
    Layers::SqrClippedReLU<L2, WeightScaleBits + 1>      ac_sqr_0;
    Layers::ClippedReLU<L2, WeightScaleBits + 1>         ac_0;
    Layers::AffineTransform<L2 * 2, FC_1_OUTPUTS>        fc_1;
    Layers::SqrClippedReLU<L3, WeightScaleBits>          ac_sqr_1;
    Layers::ClippedReLU<L3, WeightScaleBits>             ac_1;
    Layers::AffineTransform<L2 * 2 + L3 * 2, 1>          fc_2;

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

        return hashValue ^ PerspectiveHash;
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

    i32 propagate(const TransformedFeatureType* input,
                  const NNZInfo<L1>&            nnz,
                  const NetworkArchitecture&    other) const {
        struct alignas(CacheLineSize) Buffer {
            alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer first[2];
            alignas(CacheLineSize) i32 firstBoth[L2];
            alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer second[2];
            alignas(CacheLineSize) i32 secondBoth[L3];
            alignas(CacheLineSize) u8 combined[(L2 + L3) * 2];
            alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer output[2];
        };
        Buffer b;
        fc_0.propagate_pair(input, b.first[0], other.fc_0, b.first[1], nnz);
        const auto rawSkip = [](const auto& values, const auto& layer) -> i64 {
            return i64(values[FC_0_OUTPUTS - 2]) - layer.get_biases()[FC_0_OUTPUTS - 2]
                 - values[FC_0_OUTPUTS - 1] + layer.get_biases()[FC_0_OUTPUTS - 1];
        };
        const i64 skips = rawSkip(b.first[0], fc_0) + rawSkip(b.first[1], other.fc_0);
        for (int p = 0; p < 2; ++p)
            std::copy_n(b.first[p], FC_0_OUTPUTS, b.firstBoth + p * FC_0_OUTPUTS);
#if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_0.propagate_pair(b.firstBoth, b.combined, b.combined + L2);
#else
        ac_sqr_0.propagate(b.firstBoth, b.combined);
        ac_0.propagate(b.firstBoth, b.combined + L2);
#endif
        fc_1.propagate_pair(b.combined, b.second[0], other.fc_1, b.second[1]);
        for (int p = 0; p < 2; ++p)
            std::copy_n(b.second[p], FC_1_OUTPUTS, b.secondBoth + p * FC_1_OUTPUTS);
#if defined(USE_PAIR_ACTIVATIONS)
        ac_sqr_1.propagate_pair(b.secondBoth, b.combined + L2 * 2, b.combined + L2 * 2 + L3);
#else
        ac_sqr_1.propagate(b.secondBoth, b.combined + L2 * 2);
        ac_1.propagate(b.secondBoth, b.combined + L2 * 2 + L3);
#endif
        fc_2.propagate(b.combined, b.output[0]);
        other.fc_2.propagate(b.combined, b.output[1]);
        // Upstream scales: readouts and raw skips both represent 1.0 as 16384.
        // Average complete STM-score predictions once, without antisymmetry.
        const i64     merged      = (i64(b.output[0][0]) + b.output[1][0] + skips) / 2;
        constexpr i64 denominator = i64(HiddenOneVal) * (1U << WeightScaleBits) * 2;
        return i32(merged * (600 * OutputScale) / denominator);
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
