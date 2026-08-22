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
#include "clipped_relu.h"
#include "sqr_clipped_relu.h"

namespace Stockfish::Eval::NNUE::Layers {

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
    using SqrAct = SqrClippedReLU<InputDimensions, 8>;
    using LinAct = ClippedReLU<InputDimensions, 8>;

    SqrAct ac_sqr;
    LinAct ac;

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
        // 1. Add the independent bias to a temporary buffer for the squared path
        alignas(CacheLineSize) InputType lin_input[InputDimensions];
        alignas(CacheLineSize) InputType sqr_input[InputDimensions];

        for (IndexType i = 0; i < InputDimensions; ++i)
        {
            lin_input[i] = input[i];
            sqr_input[i] = input[i] + sqr_biases[i];
        }
        ac_sqr.propagate(sqr_input, output);
        ac.propagate(lin_input, output + InputDimensions);
    }

   private:
    alignas(CacheLineSize) i32 sqr_biases[InputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_DUAL_ACTIVATION_H_INCLUDED
