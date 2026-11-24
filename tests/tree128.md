# Frozen tree128 engine validation

Base: official Stockfish master `3e8fde75412e7c47fc140cd82022a395bc681d92`.
`TREE128_CONTROL` restores master's 192-stack selector and network format.
Without it the engine uses 128 frozen canonical leaf stacks. `TREE128_LAYOUT`
selects binary (0, default), conservative mixed (1), or full mixed (2).

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
and carried through export. Changing dispatch layout does not change it.

`tree128_incremental.cpp` links against matching engine objects except
`main.o`, using the **same** architecture/SIMD preprocessor flags. Run it with
a matching net and optionally an engine-roundtrip output path. It checks fresh
accumulator evaluation, incremental evaluation and undo on deterministic legal
sequences covering promotions, castling, en passant and promoted material.

For isolated ordinary PGO builds and alternating measurements:

```sh
python tests/profile_tree128.py --tree-net /path/to/tree128.nnue --control-net /path/to/master-control.nnue --build-root /path/to/new-builds
python tests/bench_tree128.py --tree-net /path/to/tree128.nnue --control-net /path/to/master-control.nnue --build-root /path/to/new-builds --output /path/to/nps.json --depth 13 --repeats 7
```

Use CPPFLAGS for selector macros: recursive PGO targets replace
EXTRACXXFLAGS. The PGO run sets the matching external EvalFile before normal
`bench`. The three tree layouts must have identical search signatures.

The local seven-repeat validation used fresh randomized networks: median NPS
binary 1,009,172; conservative mixed 1,017,512; full mixed 1,018,159; master
control 820,629. Candidate ranges overlap, so binary remains the conservative
default. The control network/search signature differs: these numbers do not
establish a trained-network speedup or chess strength. Repeat after matched
training; do not read logical dispatch-depth reductions as speedup estimates.
