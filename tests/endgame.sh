#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Compile without NDEBUG so baseline assertions remain enabled.
# Scalar or AVX2: tests/endgame.sh [scalar|avx2]
flags=()
if [[ ${1:-scalar} == avx2 ]]; then
  flags=(-mavx2 -mbmi -mpopcnt -DUSE_AVX2 -DUSE_SSE41 -DUSE_SSSE3 -DUSE_SSE2 -DUSE_POPCNT)
fi
out=${ENDGAME_TEST_OUT:-/tmp/stockfish-endgame-test}
g++ -std=c++17 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections -pthread -DIS_64BIT -DNNUE_ENDGAME_TEST "${flags[@]}" \
  tests/endgame.cpp src/attacks.cpp src/position.cpp src/bitboard.cpp src/movegen.cpp \
  src/nnue/nnue_accumulator.cpp src/nnue/features/*.cpp src/misc.cpp \
  -o "$out"
"$out"
