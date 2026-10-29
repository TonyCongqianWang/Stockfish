# Composition-conditioned material experiment

Base: `ff7cee4244ec96ea488c402669e3674c0574dce6` (freshly fetched fork default).
All baseline PSQ, full-threat, and pawn-pair features remain. L1 stays 1024;
L2, L3, layer stacks, clipping, and bilinear multiplication are unchanged.
This experimental engine requires a matching trainer export; ordinary networks
are rejected by a distinct feature hash. There is no pretrained experimental net.

## Representation

Ten bits, in order own P/N/B/R/Q then enemy P/N/B/R/Q, describe presence.
Rows are packed by increasing bucket, then relative class, omitting absent
classes. A row contributes once per piece of that class, not once per presence
bit. Both perspectives share the 5120 rows. Horizontal king mirroring does not
change their indices. Export order is biases, threats, pawn pairs, PSQ, material,
then the unchanged dense stacks. Material uses int8 weights with the existing FT
scale and int16 accumulator lane order. It receives the Packus permutation, but
not the positional paired-widening permutation: its independent kernel consumes
rows in accumulator order.

At 1024 lanes the branch has exactly 5,242,880 weights / 5,242,880 bytes (5 MiB).
Material is included before clipping and bilinear multiplication. No vector is
added at evaluation of an already computed accumulator.

## State and cache behavior

Computed stack states carry absolute piece counts. Incremental material deltas
are applied only when captures or promotions change counts. Same-bucket captures
update only changed classes. Bucket changes replace the old weighted sum with
the new one. Pawn moves, other quiet moves, and castling produce no material
state update. Null moves reuse the current stack state; undo pops it. Reverse
lazy updates invert the count delta. King refreshes use Finny entries containing
both PSQ and material; cold/stale entries reconcile their stored material counts.
The hybrid king update reconciles the old entry to the computed state's counts
so its material cancels correctly, and the new entry to the target counts.
Thus a cold/stale king cache can require material work even on a quiet king move;
a warm entry with unchanged counts does not. Accumulator width and the quiet
incremental vector operations stay unchanged; metadata checks/copies have a cost.

## Validation

Run `tests/composition.sh` and `tests/composition.sh avx2`. These standalone,
assertion-enabled tests use deterministic nonzero synthetic positional and
material weights, compare lazy incremental states against a cleared-cache full
reconstruction, and cover undo, null moves, promotions, en passant, castling,
king moves, presence-changing and same-bucket captures. Test-only counters assert
zero material state updates on quiet moves. No network download is needed.

To expose independently generated Stockfish feature IDs to trainer parity tests:

```sh
COMPOSITION_TEST_OUT=/tmp/composition-native tests/composition.sh avx2
# In nnue-pytorch, with the native loader rebuilt:
COMPOSITION_NATIVE_TEST=/tmp/composition-native pytest tests/test_composition.py
```

Build an engine for externally supplied synthetic/trained experimental networks:

```sh
make -C src -j2 all ARCH=x86-64-avx2 EXTRACXXFLAGS=-DNNUE_EMBEDDING_OFF
```

Do not use the official embedded network with this experimental format. No full
training, games, bench, or strength testing is part of correctness validation.

## Measured locally

GCC, AVX2, 1024 lanes, deterministic synthetic networks:

- Scalar and AVX2 harnesses each passed 5379 incremental/full comparisons;
  one-ply coverage included 111 quiet moves, 13 captures, 12 promotions,
  2 en passant moves, 5 castles, and 38 king moves. Captures/promotions covered
  17 presence changes and 4 changes within the same bucket; random lazy paths
  also exercised 635 null moves.
- `sizeof(FeatureTransformer)`: 135,120,896 → 140,363,776 bytes.
  `sizeof(AccumulatorState)` stays 4608 and `sizeof(Finny Entry)` stays 2176:
  the counts fit existing alignment padding on this build.
- A complete trainer export with identical 22-byte descriptions measured
  136,247,590 → 141,490,470 bytes, exactly +5 MiB (+3.85%). Trainable parameters
  measured 143,350,848 → 148,593,728, exactly +5,242,880 (+3.66%).

`tests/composition_timing.sh [baseline-checkout]` is a separate synthetic move
update microbenchmark, not engine bench or a strength test. Nine alternating
baseline/experiment runs pinned to one CPU gave these median nanoseconds per
move (includes do/undo, push/pop, accumulator evaluation, and checksum):

| Case | Baseline | Material | Difference |
| --- | ---: | ---: | ---: |
| Starting-position quiet | 733 | 717 | -2.1% |
| Middlegame quiet | 767 | 756 | -1.4% |
| Pawn-ending quiet | 426 | 430 | +0.9% |
| Middlegame king, warm cache | 1421 | 1415 | -0.4% |
| Pawn-ending king, warm cache | 416 | 420 | +0.9% |
| Middlegame capture | 1273 | 1434 | +12.6% |
| En passant, last enemy pawn | 466 | 652 | +39.9% |

Already-computed accumulator evaluation was about 12 ns in both builds, with
substantial relative timing noise at that scale. Quiet differences are within
the measurement noise; negative differences are not evidence of a speedup.
Cold/stale king cache material reconciliation has additional work. No overall
search/evaluation-throughput conclusion is justified by these small warm-cache
samples. The dense evaluation path itself is unchanged.

## Deferred material reconciliation

Spatial readiness and included material are tracked separately. Each perspective
has a count snapshot identifying the material already inside its combined
accumulator. Forward/backward spatial traversal carries that snapshot unchanged;
only the requested evaluation reconciles it with the state's actual counts.
Unevaluated intermediate buckets therefore do not construct material vectors.
Undo can reuse an intermediate spatial accumulator and reconcile its material
on demand. Hybrid king refreshes cancel the included material snapshot, which
may differ from the previous position's actual counts. Full refreshes establish
a fresh anchor. No accumulator lanes or downstream operations are added.

The harness includes three consecutive presence-bit changes without a king/H3
refresh, checks exactly one leaf reconciliation per perspective, then evaluates
each state during undo. Existing randomized lazy, refresh, and null tests remain.

## Bucket-indexed material Finny cache

Each thread has 64 tagged slots per receiving perspective (270,336 bytes total
on x86-64). High bucket bits are folded into the slot index; tags prevent aliasing.
Entries contain a material sum and the full count snapshot behind that sum.
A hit in the same composition bucket reconciles only classes whose counts differ;
an exact count match reads no material weight rows. Replacement builds the new
entry from zero. Network cache clearing invalidates all material tags.

Bucket changes reconcile the old and new endpoint cache entries and apply their
vector difference to the existing accumulator. Colliding endpoints consume the
old vector before replacing its slot. Same-bucket updates retain the direct
changed-class path. Sparse endpoints with at most four active class rows in total
also use the direct path, avoiding two int16 cache-vector reads when a few int8
rows suffice. Cold cache misses still construct material sums; usefulness depends
on reuse and is measured separately from correctness.

Tests cover warm hits with changed counts, collisions, network replacement,
reverse lazy updates and deferred leaf-only reconciliation, in scalar and AVX2.
The cache does not alter network serialization, accumulator width, downstream
layers or evaluation values, and is never added as another per-evaluation vector.
