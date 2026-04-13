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

// Definition of layer AffineTransformArgmax of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_ARGMAX_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_ARGMAX_H_INCLUDED

#include <cstdint>
#include <iostream>
#include <cstring>

#include "../../memory.h"
#include "../nnue_common.h"
#include "../simd.h"

namespace Stockfish::Eval::NNUE::Layers {

#if defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
    #define ENABLE_SEQ_OPT
#endif

template<IndexType InDims, IndexType OutDims>
class AffineTransformArgmax {
   public:
    using InputType  = std::uint8_t;
    using OutputType = std::int32_t;

    static constexpr IndexType InputDimensions  = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

#if defined(USE_AVX512)
    static constexpr IndexType ArchSimdWidth = 16;
#elif defined(USE_AVX2)
    static constexpr IndexType ArchSimdWidth = 8;
#elif defined(USE_SSSE3) || defined(USE_NEON_DOTPROD)
    static constexpr IndexType ArchSimdWidth = 4;
#else
    static constexpr IndexType ArchSimdWidth = 1;
#endif

    static constexpr bool UseSimdPath = (OutputDimensions > 1 && (OutputDimensions % ArchSimdWidth == 0));

    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash = 0x538DC7E4u) {
        std::uint32_t hashValue = 0x538DC7E4u;
        hashValue += InputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        hashValue += OutputDimensions;
        hashValue ^= prevHash >> 1;
        hashValue ^= prevHash << 31;
        return hashValue;
    }

    static constexpr IndexType get_weight_index_scrambled(IndexType i) {
        return (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4
             + i / PaddedInputDimensions * 4 + i % 4;
    }

    static constexpr IndexType get_weight_index(IndexType i) {
#ifdef ENABLE_SEQ_OPT
        if constexpr (UseSimdPath)
            return get_weight_index_scrambled(i);
        else
            return i;
#else
        return i;
#endif
    }

    bool read_parameters(std::istream& stream) {
        read_little_endian<BiasType>(stream, biases, OutputDimensions);
        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

        return !stream.fail();
    }

    bool write_parameters(std::ostream& stream) const {
        write_little_endian<BiasType>(stream, biases, OutputDimensions);

        for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
            write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

        return !stream.fail();
    }

    std::size_t get_content_hash() const {
        std::size_t h = 0;
        hash_combine(h, get_raw_data_hash(biases));
        hash_combine(h, get_raw_data_hash(weights));
        hash_combine(h, get_hash_value());
        return h;
    }

    // Forward propagation fused with argmax. Returns the bucket index.
    int propagate(const InputType* input) const {
        alignas(CacheLineSize) OutputBuffer local_output;

#ifdef ENABLE_SEQ_OPT
    // TODO move to simd.h once tested
    #if defined(USE_AVX512)
        using vec_t = __m512i;
        #define vec_set_32 _mm512_set1_epi32
        #define vec_add_32 _mm512_add_epi32
        #define vec_add_dpbusd_32 SIMD::m512_add_dpbusd_epi32
    #elif defined(USE_AVX2)
        using vec_t = __m256i;
        #define vec_set_32 _mm256_set1_epi32
        #define vec_add_32 _mm256_add_epi32
        #define vec_add_dpbusd_32 SIMD::m256_add_dpbusd_epi32
    #elif defined(USE_SSSE3)
        using vec_t = __m128i;
        #define vec_set_32 _mm_set1_epi32
        #define vec_add_dpbusd_32 SIMD::m128_add_dpbusd_epi32
    #elif defined(USE_NEON_DOTPROD)
        using vec_t = int32x4_t;
        #define vec_set_32 vdupq_n_s32
        #define vec_add_dpbusd_32(acc, a, b) \
            SIMD::dotprod_m128_add_dpbusd_epi32(acc, vreinterpretq_s8_s32(a), \
                                                vreinterpretq_s8_s32(b))
    #endif

        static constexpr IndexType OutputSimdWidth = sizeof(vec_t) / sizeof(OutputType);

        // Intelligently route to SIMD only if dimensions strictly match the architecture's width.
        if constexpr (UseSimdPath)
        {
            constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
            constexpr IndexType NumAccums = OutputDimensions / OutputSimdWidth;

    #if defined(USE_VNNI)
            constexpr IndexType NumRegs = 2 * NumAccums;
    #else
            constexpr IndexType NumRegs = NumAccums;
    #endif

            const vec_t* biasvec = reinterpret_cast<const vec_t*>(biases);
            vec_t        acc[NumRegs];
            for (IndexType k = 0; k < NumAccums; ++k)
                acc[k] = biasvec[k];
            for (IndexType k = NumAccums; k < NumRegs; ++k)
                acc[k] = vec_set_32(0);

            IndexType i = 0;
    #if defined(USE_VNNI)
            for (; i + 1 < NumChunks; i += 2)
            {
                const vec_t in0 =
                  vec_set_32(load_as<std::int32_t>(input + i * sizeof(std::int32_t)));
                const vec_t in1 =
                  vec_set_32(load_as<std::int32_t>(input + (i + 1) * sizeof(std::int32_t)));
                const auto col0 =
                  reinterpret_cast<const vec_t*>(&weights[i * OutputDimensions * 4]);
                const auto col1 =
                  reinterpret_cast<const vec_t*>(&weights[(i + 1) * OutputDimensions * 4]);

                for (IndexType k = 0; k < NumAccums; ++k)
                {
                    vec_add_dpbusd_32(acc[k], in0, col0[k]);
                    vec_add_dpbusd_32(acc[k + NumAccums], in1, col1[k]);
                }
            }
    #endif
            for (; i < NumChunks; ++i)
            {
                const vec_t in0 =
                  vec_set_32(load_as<std::int32_t>(input + i * sizeof(std::int32_t)));
                const auto col0 =
                  reinterpret_cast<const vec_t*>(&weights[i * OutputDimensions * 4]);

                for (IndexType k = 0; k < NumAccums; ++k)
                    vec_add_dpbusd_32(acc[k], in0, col0[k]);
            }

    #if defined(USE_VNNI)
            for (IndexType k = 0; k < NumAccums; ++k)
                acc[k] = vec_add_32(acc[k], acc[k + NumAccums]);
    #endif

            vec_t* outptr = reinterpret_cast<vec_t*>(local_output);
            for (IndexType k = 0; k < NumAccums; ++k)
                outptr[k] = acc[k];
        }
        else if constexpr (OutputDimensions == 1)
        {
            return 0;
        }
        else
        {
            // Bypass logic for unaligned output dimensions.
            affine_transform_non_ssse3<InputDimensions, PaddedInputDimensions, OutputDimensions>(
              local_output, weights, biases, input);
        }

#else
        affine_transform_non_ssse3<InputDimensions, PaddedInputDimensions, OutputDimensions>(
          local_output, weights, biases, input);
#endif

        int argmax_index = 0;
        OutputType max_val = local_output[0];

        for (IndexType j = 1; j < OutputDimensions; ++j) {
            if (local_output[j] > max_val) {
                max_val = local_output[j];
                argmax_index = j;
            }
        }

        return argmax_index;
    }

   private:
    using BiasType   = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
};

}  // namespace Stockfish::Eval::NNUE::Layers

#endif  // NNUE_LAYERS_AFFINE_TRANSFORM_ARGMAX_H_INCLUDED
