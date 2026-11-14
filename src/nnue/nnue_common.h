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

// Constants used in NNUE evaluation function

#ifndef NNUE_COMMON_H_INCLUDED
#define NNUE_COMMON_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "../misc.h"
#include "../types.h"

#if defined(USE_AVX2)
    #include <immintrin.h>

#elif defined(USE_SSE41)
    #include <smmintrin.h>

#elif defined(USE_SSSE3)
    #include <tmmintrin.h>

#elif defined(USE_SSE2)
    #include <emmintrin.h>

#elif defined(USE_LASX)
    #include <lasxintrin.h>
    #include <lsxintrin.h>

#elif defined(USE_LSX)
    #include <lsxintrin.h>

#elif defined(USE_NEON)
    #include <arm_neon.h>
#endif

namespace Stockfish::Eval::NNUE {

using BiasType         = i16;
using ThreatWeightType = i8;
using WeightType       = i8;
using IndexType        = u32;

// Version of the evaluation file
constexpr u32 Version = 0x6A448AFAu;

// Constant used in evaluation value calculation
constexpr int OutputScale     = 16;
constexpr int WeightScaleBits = 6;
constexpr int FtMaxVal        = 255;
constexpr int HiddenOneVal    = 128;

// Output correction fitted against nn-68752b755c93.nnue, applied to the raw
// network output before division by OutputScale. The side is selected by the
// clipped fc_0[3] activation, clamp(fc_0[3] >> 7, 0, 127) > 12. On each side the
// correction is linearly interpolated between 8 knots and constant outside them,
// which keeps the corrected output monotone in the raw output.
constexpr i32 OutputCorrectionSplit       = 13 << 7;
constexpr i32 OutputCorrectionKnots[2][8] = {
  {-15154, -9239, -6109, -3640, -38, 4134, 6721, 11166},
  {-14701, -10346, -7398, -4114, -174, 2880, 5544, 10386}};
constexpr i32 OutputCorrectionValues[2][8] = {{1782, 979, 572, 298, 105, -150, -444, -857},
                                              {2945, 1755, 1159, 604, 165, -153, -531, -1072}};

// Slope of each segment, scaled by 2^16
constexpr auto OutputCorrectionSlopes = [] {
    std::array<std::array<i32, 8>, 2> slopes{};
    for (int side = 0; side < 2; ++side)
        for (int i = 0; i < 7; ++i)
            slopes[side][i] =
              (OutputCorrectionValues[side][i + 1] - OutputCorrectionValues[side][i]) * (1 << 16)
              / (OutputCorrectionKnots[side][i + 1] - OutputCorrectionKnots[side][i]);
    return slopes;
}();

constexpr i32 correct_output(i32 raw, i32 fc0_3) {
    const int   side  = fc0_3 >= OutputCorrectionSplit;
    const auto& knots = OutputCorrectionKnots[side];
    const i32   x     = std::clamp(raw, knots[0], knots[7]);
    int         seg   = 0;
    for (int i = 1; i < 8; ++i)
        seg += knots[i] < x;
    const i32 correction = OutputCorrectionValues[side][seg]
                         + (((x - knots[seg]) * OutputCorrectionSlopes[side][seg]) >> 16);
    return std::clamp(raw + correction, -512000, 512000);
}

// Size of cache line (in bytes)
constexpr usize CacheLineSize = 64;

// clang-format off
// Orient a square according to perspective
// Reflect kings on files a..d, so that it lands on e..h
alignas(CacheLineSize) inline constexpr u8 OrientTBL[SQUARE_NB] = {
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
    SQ_H1, SQ_H1, SQ_H1, SQ_H1, SQ_A1, SQ_A1, SQ_A1, SQ_A1,
};
// clang-format on


constexpr usize MaxSimdWidth = 32;

// Type of input feature after conversion
using TransformedFeatureType = u8;

// Round n up to be a multiple of base
template<typename IntType>
constexpr IntType ceil_to_multiple(IntType n, IntType base) {
    return (n + base - 1) / base * base;
}


// Utility to read an integer (signed or unsigned, any size)
// from a stream in little-endian order. We swap the byte order after the read if
// necessary to return a result with the byte ordering of the compiling machine.
template<typename IntType>
inline IntType read_little_endian(std::istream& stream) {
    IntType result;

    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
    else
    {
        u8                            u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = 0;

        stream.read(reinterpret_cast<char*>(u), sizeof(IntType));
        for (usize i = 0; i < sizeof(IntType); ++i)
            v = (v << 8) | u[sizeof(IntType) - i - 1];

        std::memcpy(&result, &v, sizeof(IntType));
    }

    return result;
}


// Utility to write an integer (signed or unsigned, any size)
// to a stream in little-endian order. We swap the byte order before the write if
// necessary to always write in little-endian order, independently of the byte
// ordering of the compiling machine.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, IntType value) {

    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(&value), sizeof(IntType));
    else
    {
        u8                            u[sizeof(IntType)];
        std::make_unsigned_t<IntType> v = value;

        usize i = 0;
        // if constexpr to silence the warning about shift by 8
        if constexpr (sizeof(IntType) > 1)
        {
            for (; i + 1 < sizeof(IntType); ++i)
            {
                u[i] = u8(v);
                v >>= 8;
            }
        }
        u[i] = u8(v);

        stream.write(reinterpret_cast<char*>(u), sizeof(IntType));
    }
}


// Read integers in bulk from a little-endian stream.
// This reads N integers from stream s and puts them in array out.
template<typename IntType>
inline void read_little_endian(std::istream& stream, IntType* out, usize count) {
    if (IsLittleEndian)
        stream.read(reinterpret_cast<char*>(out), sizeof(IntType) * count);
    else
        for (usize i = 0; i < count; ++i)
            out[i] = read_little_endian<IntType>(stream);
}


// Write integers in bulk to a little-endian stream.
// This takes N integers from array values and writes them on stream s.
template<typename IntType>
inline void write_little_endian(std::ostream& stream, const IntType* values, usize count) {
    if (IsLittleEndian)
        stream.write(reinterpret_cast<const char*>(values), sizeof(IntType) * count);
    else
        for (usize i = 0; i < count; ++i)
            write_little_endian<IntType>(stream, values[i]);
}


}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_COMMON_H_INCLUDED
