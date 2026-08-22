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

#include "features/full_threats.h"
#include "features/half_ka_v2_hm.h"
#include "features/pp_3wide.h"
#include "layers/affine_transform.h"
#include "layers/affine_transform_i16_i8.h"
#include "layers/affine_transform_sparse_input.h"
#include "layers/dual_activation.h"
#include "layers/inverted_bottleneck_block.h"
#include "nnue_common.h"
#include "nnz_helper.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using ThreatFeatureSet = Features::FullThreats;
using PairFeatureSet   = Features::PP_3Wide;
using PSQFeatureSet    = Features::HalfKAv2_hm;

// Model architecture configuration (matching nnue-pytorch LayerStacksConfig)
constexpr IndexType L1                    = 1024;
constexpr int       ResDim                = 32;  // Residual stream dimension (res_dim)
constexpr int       ExpandedDim           = 64;  // Expanded dimension inside bottleneck blocks (expanded_dim)
constexpr int       NumBlocks             = 3;   // Total number of inverted bottleneck blocks (num_blocks)
constexpr int       NumIntermediateBlocks = NumBlocks - 1;

// Compatibility aliases
constexpr int L2 = ResDim;
constexpr int L3 = ExpandedDim / 2;

constexpr IndexType PSQTBuckets = 8;
constexpr IndexType LayerStacks = 8;

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
static_assert(
  PSQTBuckets % 8 == 0,
  "Per feature PSQT values cannot be processed at granularity lower than 8 at a time.");

struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       ResDim                       = Stockfish::Eval::NNUE::ResDim;
    static constexpr int       ExpandedDim                  = Stockfish::Eval::NNUE::ExpandedDim;
    static constexpr int       FC_0_OUTPUTS                 = ResDim;
    static constexpr int       FC_1_OUTPUTS                 = ExpandedDim / 2;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, ResDim> l1;
    Layers::InvertedBottleneckBlock<ResDim, ExpandedDim, false> blocks[NumIntermediateBlocks];
    Layers::InvertedBottleneckBlock<ResDim, ExpandedDim, true>  final_block;

    // Hash value embedded in the evaluation file
    static constexpr u32 get_hash_value() {
        u32 hashValue = 0xEC42E90Du ^ (TransformedFeatureDimensions * 2);

        // 1. l1 layer
        u32 h_l1 = 0xCC03DAE4u + ResDim;
        h_l1 ^= hashValue >> 1;
        h_l1 ^= hashValue << 31;
        h_l1 += 0x538D24C7u;
        hashValue = h_l1;

        // 2. Intermediate blocks
        for (int i = 0; i < NumIntermediateBlocks; ++i)
        {
            u32 h_up = 0xCC03DAE4u + ExpandedDim;
            h_up ^= hashValue >> 1;
            h_up ^= hashValue << 31;
            h_up += 0x538D24C7u;
            hashValue = h_up;

            u32 h_down = 0xCC03DAE4u + ResDim;
            h_down ^= hashValue >> 1;
            h_down ^= hashValue << 31;
            h_down += 0x538D24C7u;
            hashValue = h_down;
        }

        // 3. Final block up layer
        u32 h_fup = 0xCC03DAE4u + ExpandedDim;
        h_fup ^= hashValue >> 1;
        h_fup ^= hashValue << 31;
        h_fup += 0x538D24C7u;
        hashValue = h_fup;

        // 4. Final block output layer (out_features = 1)
        u32 h_out = 0xCC03DAE4u + 1;
        h_out ^= hashValue >> 1;
        h_out ^= hashValue << 31;
        hashValue = h_out;

        return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
        if (!l1.read_parameters(stream))
            return false;
        for (int i = 0; i < NumIntermediateBlocks; ++i)
            if (!blocks[i].read_parameters(stream))
                return false;
        return final_block.read_parameters(stream);
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
        if (!l1.write_parameters(stream))
            return false;
        for (int i = 0; i < NumIntermediateBlocks; ++i)
            if (!blocks[i].write_parameters(stream))
                return false;
        return final_block.write_parameters(stream);
    }

    i32 propagate(const TransformedFeatureType* transformedFeatures,
                  const NNZInfo<L1>&            nnzInfo) const {
        // 1. Initial linear projection l1 -> l1_out preactivations
        typename decltype(l1)::OutputBuffer l1_out;
        l1.propagate(transformedFeatures, l1_out, nnzInfo);

        // 2. Initial Residual Stream R in int32_t (scale 1 << ResQuantizedOneBits)
        alignas(CacheLineSize) i32 res_stream[ResDim];
        for (int i = 0; i < ResDim; ++i)
            res_stream[i] = l1_out[i] >> InferenceL1Shift;

        // 3. Intermediate bottleneck residual blocks
        for (int i = 0; i < NumIntermediateBlocks; ++i)
            blocks[i].propagate(res_stream);

        // 4. Final bottleneck block -> fused output preactivation
        i32 fwdOut = final_block.propagate(res_stream);

        // 5. Convert to internal score units do not simplify formula as it corresponds to pytorch values.
        constexpr int OutputDivisorBits = ResQuantizedOneBits + WeightScaleOutResBits;
        static_assert(OutputDivisorBits >= 0 && OutputDivisorBits < 31,
                      "OutputDivisorBits must be in [0, 30] so (1 << OutputDivisorBits) fits in signed 32-bit");
        i32 outputValue = static_cast<i32>((static_cast<i64>(fwdOut) * NNUE2Score * OutputScale)
                                           / (1 << OutputDivisorBits));
        return outputValue;
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, l1.get_content_hash());
        for (int i = 0; i < NumIntermediateBlocks; ++i)
            hash_combine(h, blocks[i].get_content_hash());
        hash_combine(h, final_block.get_content_hash());
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
