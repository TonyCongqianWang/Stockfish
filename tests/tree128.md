# Frozen tree128 engine validation

Base: official Stockfish master `3e8fde75412e7c47fc140cd82022a395bc681d92`.
The engine uses one simplified binary evaluator and 128 frozen canonical
leaf stacks. The master control uses an unmodified upstream engine checkout.
There are no alternative layout or control-build switches.

The generated rules and shared lazy extractor come from the trainer:

```sh
python -m routing.tree128.generate --engine /path/to/Stockfish
```

Build a plain external-network engine without downloading an embedded net:

```sh
make -C src -j4 -o net build ARCH=x86-64-avx2 CPPFLAGS=-DNNUE_EMBEDDING_OFF
```

Use UCI `setoption name EvalFile value /path/to/matching-tree128.nnue` before
evaluation/search. `tree128` prints fields and leaf for extraction parity
without loading a network. The full semantic SHA256 is checked when loading
and carried through export. The semantic identity is independent of generated code formatting.

`tree128_incremental.cpp` links against matching engine objects except
`main.o`, using the **same** architecture/SIMD preprocessor flags. Run it with
a matching net and optionally an engine-roundtrip output path. It checks fresh
accumulator evaluation, incremental evaluation and undo on deterministic legal
sequences covering promotions, castling, en passant and promoted material.

For PGO, use Stockfish's ordinary `profile-build` target. Set its existing
`PGOBENCH` variable to a shell script that selects the matching 128-stack
EvalFile and runs normal `bench`:

```sh
# /path/to/profile-tree128.sh (invoked with src as its working directory)
printf 'setoption name EvalFile value /path/to/tree128.nnue\nbench\nquit\n' | ./stockfish

make -C src -j4 profile-build ARCH=native PGOBENCH='sh /path/to/profile-tree128.sh'
```

The earlier layout comparison found no clear gain beyond run variation.
Only the simplified binary evaluator is retained. Historical measurements
remain in the trainer validation report; alternative implementations and
their build/benchmark machinery have been removed.
