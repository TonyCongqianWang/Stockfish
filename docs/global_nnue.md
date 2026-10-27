# Experimental mirrored NNUE architecture

This branch requires newly trained networks. Previous experimental networks,
upstream master networks and incompatible checkpoints are rejected. Test
fixtures are untrained and are not playing-strength baselines.

The feature contract, in serialization order, is:

| Feature | Rows | Columns | Hash |
| --- | ---: | ---: | --- |
| Own_Supports | 14,952 | 768 | 0x4f535031 |
| Enemy_Supports | 14,952 | 256 | 0x45535031 |
| Global_Attacks | 29,904 | 1,024 | 0x47415432 |
| Pawn_FrontierPairs | 16,807 | 1,024 | 0x50465032 |
| Pawn_FilePairs | 480 | 1,024 | 0x50464332 |
| K32Q2 | 45,056 | 1,024 | 0x32b5e284 |

Format version is `0x6a448afc`. A shared 1,024-element int16 bias precedes
these int8 tables, followed by the unchanged 32 downstream layer stacks.
Feature hashes combine by rotate-left-one and XOR in table order.

Each perspective retains 1,024 raw lanes. The first and second 512-lane
operands multiply after clipping, giving 512 outputs per perspective and 1,024
combined downstream inputs. Product indices 0..383 receive own support in both
operands; indices 384..511 receive enemy support in both operands. PSQ, attacks
and pawn inputs contribute throughout. There is no unpaired section.

Four aligned int16 accumulator banks contain two perspective-specific base
sums (bias, PSQ, attacks, pawns) and two owner-associated support sums. Each
support bank packs 768 owner-facing and 256 opponent-facing lanes. A support
change updates 1,024 lanes; an attack change can update both base banks, totaling
2,048 lanes. Support tables have independent learned weights, shared across
colors, with no negative signs or tied positive/negative virtual rows.

All spatial features use the receiving perspective's PSQ orientation: reflect
ranks and swap colors for Black, and reflect files when that perspective's king
is on files a..d. Own-facing support uses its owner's orientation; opponent-facing
support uses the other king's orientation. Each section is independently lazy.
Horizontal king crossings are refresh barriers for every section feeding that
perspective. PSQ Finny caches remain PSQ-only. Compact support weights/sums are
stored in canonical lane order; activation assembly applies SIMD packing.

Threat relations retain upstream's target exclusions and equal-type contact
deduplication. Kings are excluded. Normalize the endpoints before canonicalizing
equal-type contacts. Per-color geometry bases are 0,168,1848,4088,7672 for
P,N,B,R,Q, with strides 84,336,560,896,1456.

Pawn frontiers use missing=0 and ranks 2..7=1..6. On a file, `s=7*w+b`, where
White's frontier is its highest pawn and Black's is its lowest pawn. Boundary
`f` uses `2401*f+49*s[f]+s[f+1]`, including empty files. Black normalization
swaps colors and maps nonzero ranks q to 7-q. Horizontal mirroring maps a
boundary to 6-f and swaps its file states. Within-file pairs activate only for
consecutive occupied pawn ranks, regardless of color. Their vocabulary is 15
White-White, 15 Black-Black, then 30 White-Black distinct-rank states in
lexicographic order; mirroring maps their file to 7-f. The two pawn families
together preserve full pawn placement.

Training-free validation:

```sh
cd src
make -j2 stockfish global-nnue-test ARCH=x86-64-avx2 EXTRACXXFLAGS=-DNNUE_EMBEDDING_OFF
./global-nnue-test
./global-nnue-test /absolute/path/to/positions.epd
./global-nnue-test --dump /absolute/path/to/positions.epd
```

The harness checks incremental/full sums and activation assembly, undo/null,
deferred updates, refresh barriers, promotions, en passant and overlapping or
stationary-king Chess960 castling. Position-file entries can append `chess960`
after the six FEN fields. `tests/generate_global_nnue_fixture.py output.nnue`
generates a stdlib-only external EvalFile fixture. Scalar/UBSan and AVX2 checks
run in branch CI. Legacy bench/signature, PGO, game and release jobs remain
gated while there is no compatible trained default network.
