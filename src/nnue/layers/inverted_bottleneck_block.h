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

// Definition of layer InvertedBottleneckBlock of NNUE evaluation function

#ifndef NNUE_LAYERS_INVERTED_BOTTLENECK_BLOCK_H_INCLUDED
#define NNUE_LAYERS_INVERTED_BOTTLENECK_BLOCK_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <tuple>
#include <type_traits>

#include "../nnue_common.h"
#include "affine_transform.h"
#include "affine_transform_i16_i8.h"
#include "dual_activation.h"

namespace Stockfish::Eval::NNUE::Layers {

template<IndexType ResDim = 32, IndexType ExpandedDim = 64, bool IsFinalBlock = false>
class InvertedBottleneckBlock {
   public:
    static constexpr IndexType InputDimensions       = ResDim;
    static constexpr IndexType OutputDimensions      = IsFinalBlock ? 1 : ResDim;
    static constexpr IndexType ActDimensions         = ExpandedDim * 2;
    static constexpr IndexType FusedInputDimensions  = ResDim + ActDimensions;
    static constexpr IndexType PaddedFusedDimensions =
      ceil_to_multiple<IndexType>(FusedInputDimensions, 32);

    static constexpr u32 get_hash_value(u32 prevHash) {
        u32 hashValue = decltype(up)::get_hash_value(prevHash);
        if constexpr (!IsFinalBlock)
        {
            hashValue = decltype(down)::get_hash_value(hashValue);
        }
        else
        {
            u32 outHash = 0xCC03DAE4u + 1;
            outHash ^= hashValue >> 1;
            outHash ^= hashValue << 31;
            hashValue = outHash;
        }
        return hashValue;
    }

    bool read_parameters(std::istream& stream) {
        if (!up.read_parameters(stream) || !act.read_parameters(stream))
            return false;

        if constexpr (!IsFinalBlock)
        {
            return down.read_parameters(stream);
        }
        else
        {
            read_little_endian<i32>(stream, &output_bias, 1);
            for (IndexType i = 0; i < PaddedFusedDimensions; ++i)
                output_weights[i] = read_little_endian<i8>(stream);
            return !stream.fail();
        }
    }

    bool write_parameters(std::ostream& stream) const {
        if (!up.write_parameters(stream) || !act.write_parameters(stream))
            return false;

        if constexpr (!IsFinalBlock)
        {
            return down.write_parameters(stream);
        }
        else
        {
            write_little_endian<i32>(stream, &output_bias, 1);
            for (IndexType i = 0; i < PaddedFusedDimensions; ++i)
                write_little_endian<i8>(stream, output_weights[i]);
            return !stream.fail();
        }
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, up.get_content_hash());
        hash_combine(h, act.get_content_hash());
        if constexpr (!IsFinalBlock)
        {
            hash_combine(h, down.get_content_hash());
        }
        else
        {
            hash_combine(h, get_raw_data_hash(output_bias));
            hash_combine(h, get_raw_data_hash(output_weights));
        }
        hash_combine(h, get_hash_value(0));
        return h;
    }

    i32 propagate(i32* res_stream) const {
        // 1. Pre-block clamp R in-place to [-32767, 32767]
        i16 clamped_r[ResDim];
        for (IndexType i = 0; i < ResDim; ++i)
            clamped_r[i] =
              static_cast<i16>(std::clamp(res_stream[i], -32767, 32767));

        // 2. Up-Projection (i16 -> i32 preactivation)
        typename AffineTransformI16I8<ResDim, ExpandedDim>::OutputBuffer up_out;
        up.propagate(clamped_r, up_out);

        // 3. DualActivation (i32 preactivation -> u8 expanded features)
        typename DualActivation<ExpandedDim>::OutputBuffer act_out;
        act.propagate(up_out, act_out);

        if constexpr (!IsFinalBlock)
        {
            // 4. Down-Projection
            typename AffineTransform<ActDimensions, ResDim>::OutputBuffer down_out;
            down.propagate(act_out, down_out);

            // 5. Standard Residual Addition: res_stream += (down_out >> 6)
            for (IndexType i = 0; i < ResDim; ++i)
            {
                i32 delta     = down_out[i] >> 6;
                res_stream[i] = res_stream[i] + delta;
            }
        }
        else
        {
            // 4. Fused Output Projection
            i32 sum = output_bias;
            for (IndexType j = 0; j < ResDim; ++j)
                sum += static_cast<i32>(clamped_r[j]) * static_cast<i32>(output_weights[j]);
            for (IndexType k = 0; k < ActDimensions; ++k)
                sum += 2 * (static_cast<i32>(act_out[k])
                            * static_cast<i32>(output_weights[ResDim + k]));

            return sum;
        }
        return 0;
    }

   private:
    AffineTransformI16I8<ResDim, ExpandedDim> up;
    DualActivation<ExpandedDim>               act;

    // Active for intermediate block
    std::conditional_t<!IsFinalBlock, AffineTransform<ActDimensions, ResDim>, std::tuple<>> down;

    // Active for final block
    alignas(CacheLineSize) std::conditional_t<IsFinalBlock, i32, std::tuple<>> output_bias;
    alignas(CacheLineSize) std::conditional_t<IsFinalBlock, i8[PaddedFusedDimensions], std::tuple<>>
      output_weights;
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_INVERTED_BOTTLENECK_BLOCK_H_INCLUDED
