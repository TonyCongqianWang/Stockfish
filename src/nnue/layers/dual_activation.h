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

// Definition of layer DualActivation of NNUE evaluation function

#ifndef NNUE_LAYERS_DUAL_ACTIVATION_H_INCLUDED
#define NNUE_LAYERS_DUAL_ACTIVATION_H_INCLUDED

#include <algorithm>
#include <cstdint>
#include <iosfwd>

#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

// Elementwise activation helpers
inline u8 square_clipped_relu(i32 input, i32 sqr_bias) {
    i64 sum = static_cast<i64>(input) + 2 * static_cast<i64>(sqr_bias);
    if (sum <= 0) return 0;
    i32 sqr_val = static_cast<i32>((sum * sum) >> 23);
    return static_cast<u8>(std::min(sqr_val, 127));
}

inline u8 clipped_relu(i32 input) {
    if (input <= 0) return 0;
    i32 lin_val = input >> 8;
    return static_cast<u8>(std::min(lin_val, 127));
}

template<IndexType InDims>
class DualActivation {
   public:
    using InputType  = i32;
    using OutputType = u8;

    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions * 2;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    static constexpr u32 get_hash_value(u32 prevHash) {
        u32 hashValue = 0x538D24C7u;
        hashValue += prevHash;
        return hashValue;
    }

    bool read_parameters(std::istream& stream) {
        read_little_endian<i32>(stream, sqr_biases, InputDimensions);
        return !stream.fail();
    }

    bool write_parameters(std::ostream& stream) const {
        write_little_endian<i32>(stream, sqr_biases, InputDimensions);
        return !stream.fail();
    }

    usize get_content_hash() const {
        usize h = 0;
        hash_combine(h, get_raw_data_hash(sqr_biases));
        hash_combine(h, get_hash_value(0));
        return h;
    }

    void propagate(const InputType* input, OutputType* output) const {
        for (IndexType i = 0; i < InputDimensions; ++i)
        {
            output[i]                   = square_clipped_relu(input[i], sqr_biases[i]);
            output[InputDimensions + i] = clipped_relu(input[i]);
        }
    }

   private:
    alignas(CacheLineSize) i32 sqr_biases[InputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_DUAL_ACTIVATION_H_INCLUDED
