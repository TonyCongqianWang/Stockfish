# Agreement with recorded LTC moves

The initial nnue-tools benchmark link contains sampled quiet positions and scores, not played moves or complete games. This test uses the complete PGN archive from the subsequently supplied Fishtest run:

https://tests.stockfishchess.org/tests/view/6af233258e7fb2e78bc1cc0b

The archive was successfully downloaded from `/api/run_pgns/6af233258e7fb2e78bc1cc0b.pgn.gz`. It contains 59,974 PGNs. Its SHA-256 and the selected source indices are in `sample-manifest.json`. The included `sample.pgn` preserves the exact 16-game sample, so reproduction does not depend on continued PGN availability.

## Protocol

- Uniform reservoir sample of 16 complete games, seed 20261116, chosen before searching.
- 2,423 recorded moves, including captures, checks, promotions, and endgames. Of these, 2,404 positions have more than one legal move.
- A: original master `5d03b6e5`, B: removal-only, C: unchanged fuzzy patch `d8b70279`.
- One thread, MultiPV=1, Hash=32 MiB; identical network and release builds from the original experiment. No tablebases loaded.
- Separate runs at 10,000 and 100,000 nodes per move. A budget is a UCI node limit, with the engine's usual checking granularity.
- Two independent engine processes per game replay, one per color. Each retains its own TT and histories across its turns. Both are reset between games. Variants and budgets have independent state.
- Before every search, send the original starting FEN plus the entire recorded move prefix. Search all legal root moves, record the engine's unrestricted choice, then **advance with the recorded PGN move**. The candidate's choice never changes a subsequent position.
- Assert that every candidate choice is legal and that the advanced board matches the PGN node after every move. All games are replayed to the end of their recorded mainline, including after an earlier mismatch.
- 14,538 searches in total. The engine patch was not modified.

PGNs start at the post-book FEN supplied by Fishtest; all history recorded after that point is preserved. The pre-book history is absent from the PGN. Final bound flags are parsed from the last score-bearing UCI line, rather than accumulated across info lines.

The teacher run used eight threads, Hash=512 MiB, and nominal LTC 60+0.6 (individual PGN clocks are CPU-scaled). It compared Stockfish 19 (`edb0d9db...`, net `nn-1a298aa575a0`) with master `43e01e07...` (net `nn-18cd32e11a9c`). Student A/B/C use a later master and `nn-bddd9d57eb87`. Thus this tests agreement with independent, much larger searches; it is not same-version ground truth. Equal-strength moves, draw choices, SMP variability, network changes, and the source's possible tablebase use can all cause harmless mismatches.

## Results

| Nodes per move | A | B | C | C minus A | Paired game-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| 10,000 | 59.60% | 59.93% | 60.26% | +0.66 pp | [-0.49, +2.11] pp |
| 100,000 | 68.72% | 69.17% | 69.01% | +0.29 pp | [-1.19, +1.83] pp |

First 40 replayed plies per game (640 nonforced positions):

| Nodes per move | A | B | C |
|---|---:|---:|---:|---:|
| 10,000 | 61.41% | 63.12% | 61.25% |
| 100,000 | 72.19% | 73.91% | 74.38% |

Nonforced positions only (2,404 positions):

| Nodes per move | A | B | C |
|---|---:|---:|---:|---:|
| 10,000 | 59.28% | 59.61% | 59.94% |
| 100,000 | 68.47% | 68.93% | 68.76% |

Teacher matches by count and changes relative to A:

- 10,000 nodes: C 1460/2,423 versus A 1444/2,423. Of 768 differing choices, C alone matched 248, A alone matched 232.
- 100,000 nodes: C 1672/2,423 versus A 1665/2,423. Of 622 differing choices, C alone matched 228, A alone matched 221.

All 14,538 searches completed. A separate fresh-process repeat of C at 10k nodes reproduced all 201 choices, reported node counts, depths, and bound flags in the first sampled game. Every chosen move was legal, every recorded continuation was preserved, and every final JSONL file contains exactly the same 2,423 position/label keys. Exact counts, phase and teacher splits, equal-weighted game averages, reported bound frequencies, and paired intervals versus both A and B are in `summary.json`. Final bound flags include ordinary interrupted searches in all variants, not just C’s accepted fuzzy iterations.

C versus A and C versus B game-bootstrap intervals include zero at both budgets. This sample does not distinguish the variants. The aggregate differences are small compared with the effect of increasing the search budget. Inspect the early-game subset and game-level intervals before treating an overall rate as evidence of improvement. This is a useful fixed-node move-selection proxy; it does not test timed allocation or establish playing-strength equivalence.

The primary rate is exact recorded-move agreement, weighted by moves. The analysis also reports equal-weighted game means, forced-move exclusions, game phase, teacher version, and the first 40 recorded plies of each game. The first-40 subset has 640 positions, all nonforced, and reduces the influence of long endgames.

Paired differences compare the same positions. Their descriptive 95% percentile intervals use 10,000 bootstrap resamples of whole games (seed 20261116), rather than treating moves from one game as independent observations. Sixteen game clusters are still a small sample. Exact-move mismatches are not necessarily mistakes, and agreement differences do not establish an Elo difference.

## Reproduction

Requires python-chess and the A/B/C release binaries from `../build.py`. The supplied sample can be replayed directly:

```sh
python3 experiments/fuzzy-score/ltc-replay/replay_pgn.py \
  --pgn experiments/fuzzy-score/ltc-replay/sample.pgn \
  --bin-dir /path/to/fuzzy-builds --out /tmp/ltc-replay \
  --budgets 10000 100000 --jobs 2
python3 experiments/fuzzy-score/ltc-replay/compare_replay.py \
  --results /tmp/ltc-replay --out /tmp/ltc-replay-summary.json
```

To select a new sample from a downloaded complete archive:

```sh
python3 experiments/fuzzy-score/ltc-replay/sample_pgn.py \
  --source /path/to/games.pgn.gz --out /tmp/ltc-sample \
  --games 16 --seed 20261116
```

The delivered `ltc-replay-results.zip` contains the raw per-position JSONL records, sample PGNs, source metadata, scripts, and summary.
