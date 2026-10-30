#!/usr/bin/env bash
# Run from src/ after an AVX2 build; keep the test executable out of the build tree.
set -euo pipefail
binary=$(mktemp "${TMPDIR:-/tmp}/advanced-pawn-pairs.XXXXXX")
trap 'rm -f "$binary"' EXIT
objects=$(make --no-print-directory --eval='print-test-objects:;@echo $(filter-out main.o,$(OBJS))' print-test-objects)
# Object names come from the Makefile, avoiding stale objects from other architectures/features.
# shellcheck disable=SC2086
"${CXX:-g++}" -std=c++17 -O2 -DUSE_AVX2 -DUSE_POPCNT -DIS_64BIT \
    -mavx2 -mbmi -mpopcnt ../tests/advanced_pawn_pairs.cpp $objects \
    -lpthread -lrt -o "$binary"
"$binary"
