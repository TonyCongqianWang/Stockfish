#!/usr/bin/env bash
# Synthetic accumulator microbenchmark. Does not perform search or play games.
set -euo pipefail
source_root=$(cd "$(dirname "$0")/.." && pwd)
engine_root=${1:-$source_root}
flags=()
if [[ -f $engine_root/src/nnue/features/composition.h ]]; then
  flags=(-DMATERIAL)
fi
out=${COMPOSITION_TIMING_OUT:-/tmp/composition-timing}
cd "$engine_root"
g++ -std=c++17 -O3 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -pthread -DIS_64BIT -mavx2 -mbmi -mpopcnt -DUSE_AVX2 -DUSE_SSE41 \
  -DUSE_SSSE3 -DUSE_SSE2 -DUSE_POPCNT "${flags[@]}" -Isrc \
  "$source_root/tests/composition_timing.cpp" src/attacks.cpp src/position.cpp \
  src/bitboard.cpp src/movegen.cpp src/nnue/nnue_accumulator.cpp \
  src/nnue/features/*.cpp src/misc.cpp -o "$out"
"$out"
