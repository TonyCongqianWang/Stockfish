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

// Definition of layer AffineTransformI16I8 of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_I16_I8_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_I16_I8_H_INCLUDED

#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

// AffineTransformI16I8 multiplies 16-bit signed inputs (scale 256.0)
// by 8-bit signed weights (scale 256.0), shifts down to scale 32768.0,
// and adds 32-bit signed biases (scale 32768.0).
template<IndexType InputDimensions, IndexType OutputDimensions>
class AffineTransformI16I8 {
   public:
    using InputType  = i16;
    using OutputType = i32;
    using BiasType   = i32;
    using WeightType = i8;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, 32);

    using OutputBuffer = OutputType[OutputDimensions];

    static constexpr u32 get_hash_value(u32 prevHash) {
        u32 hashValue = 0xCC03DAE4u;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        if constexpr (OutputDimensions != 1)
            hashValue += 0x538D24C7u;
        return hashValue;
    }

    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            weights[i] = read_little_endian<WeightType>(stream);
        return !stream.fail();
    }

    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            write_little_endian<WeightType>(stream, weights[i]);
        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    void propagate(const InputType* input, OutputType* output) const {
        for (IndexType i = 0; i < OutputDimensions; ++i)
        {
            i32 product_sum = 0;
            const WeightType* row = &weights[i * PaddedInputDimensions];
            for (IndexType j = 0; j < InputDimensions; ++j)
            {
                product_sum += static_cast<i32>(input[j]) * static_cast<i32>(row[j]);
            }
            output[i] = (product_sum + biases[i]) >> 1;
        }
    }

   private:
    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_AFFINE_TRANSFORM_I16_I8_H_INCLUDED
