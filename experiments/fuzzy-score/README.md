# Fuzzy aspiration score experiment

Engine patch: `d8b7027952c15ca02c15b40110f44ffdafd4e0cb` (39 insertions, 16 deletions; two source files).
Base: TonyCongqianWang/Stockfish master `5d03b6e578a5eedc822e4563bdbb89e762fc2a59`, verified against the remote on 2026-11-16.
Work was done in a separate local clone; the supplied checkout was left untouched.

This is an experimental patch, not an established strength improvement.

## Behavior

A is the original engine, including `failHighRecovery`. B removes only that mechanism. C removes it and allows a completed lower-bound iteration after **two consecutive fail-highs**. The second attempt still has the existing `failedHighCnt` depth reduction. Ordinary fail-lows still reset that count and widen the window, exactly as before. No full-depth widening replacement was introduced.

The existing RootMove bound flags, PV, UCI threshold, and TT entries already hold most of the required information. C adds only `Worker::fuzzyScore` and `RootMove::effectiveDepth`. It retains the move and PV, stores the established beta threshold as the lower-bound score, and reports its actual root depth. It also preserves the bound when setting `previousScoreExact` for the next iteration.

For eligible non-mate searches, observations entering both `averageScore` and `meanSquaredScore` are capped at beta on fail-highs, including ordinary attempts. The existing weights are unchanged. Fail-low observations are unchanged. `meanSquaredScore` is a signed second moment, not a prediction-error variance; it is not repurposed as a confidence model. A final interrupted non-decisive fail-high also stores beta instead of its fail-soft overshoot. The falling-evaluation time factor cannot claim an improvement from a lower bound: its observation is capped at both comparison scores. Existing effort and PV-stability time factors are unchanged; repeated cutoffs can still make the PV appear dominant and stable, so this is not confidence-aware time allocation.

Eligibility is single thread, one effective PV, full skill, no mate limit, no root tablebase result, and standard chess. Mate-seeking iterations and decisive values keep the ordinary aspiration loop and predictor observations. MultiPV, multi-threaded search, handicapped play, root tablebases, and Chess960 follow B rather than fuzzy termination. Removing recovery necessarily changes those modes relative to A.

## What the code requires from the previous iteration

An exact previous score is not a prerequisite for the next search. Aspiration starts from the move's moving average and signed second moment, the previous PV supplies the root TT move and helps PV ordering, and TT entries carry their own bounds and depths. `previousScore` also influences tie-breaking and mate seeking. Bounds must not be relabeled exact for the existing MultiPV fallback path.

Shallow fail-soft results can inflate the old predictor and optimism. C removes their direct upward contribution. It deliberately preserves ordinary aspiration widening (`beta = bestValue + delta`), so a later threshold can still originate from a previous fail-soft result; accepting it requires another successful fail-high. Clipping is not a proof of stability across depths.

A fail-high can terminate on the first root move. Finishing an iteration this way does **not** establish that competitors are worse. It postpones those comparisons. The next iteration starts at a greater requested depth than the accepted reduced-depth attempt, but can again cut off on that PV.

Within ordinary aspiration, a reduced-depth fail-low invalidates the optimistic window, resets `failedHighCnt`, and resumes the deeper search. C keeps that behavior. It does not keep an earlier deeper bound as a replacement for a later fail-low. Such a retention policy would be a second behavioral experiment, particularly relevant to an optional final verification search.

Thread voting normally uses raw scores, while decisive votes require exact flags. Enabling this experiment in multiple threads without changing voting would mix bound and exact estimates. C avoids that issue by disabling fuzzy behavior in multi-threaded mode; voting is unchanged.

## Position comparisons

One thread, MultiPV=1, Hash=32 MiB, default verified network `nn-bddd9d57eb87.nnue`, GCC, AVX2, identical non-PGO build options. The deterministic corpus contains 49 standard bench positions and 80 sampled positions from the built-in benchmark games. There are 129 positions at depth 16 and at 300,000 nodes, plus 16 positions at depth 20: **822 searches** across A/B/C. PVs and final moves were checked for legality with python-chess.

| Suite | A nodes | B nodes | C nodes | C vs A | C vs B |
|---|---:|---:|---:|---:|---:|
| Standard bench | 1,442,480 | 2,281,092 | 1,189,342 | -17.5% | -47.9% |
| 129 positions, depth 16 | 15,805,682 | 22,096,649 | 11,456,286 | -27.5% | -48.2% |
| 16 positions, depth 20 | 11,004,875 | 10,706,980 | 5,773,155 | -47.5% | -46.1% |
| 129 positions, 300k nodes | 38,122,782 | 38,123,124 | 38,122,279 | budget | budget |

After the timed matches, three uninstrumented bench repetitions in rotated A/B/C order gave median engine-reported search times A=1,797 ms, B=2,818 ms, C=1,565 ms (C 12.9% below A). Whole-process startup times were noisy; these small local timing samples are not a throughput guarantee. All nine repetitions reproduced their signatures. `results/wall-bench.json` includes both search and whole-process times.

The node-limited corpus includes terminal positions, so aggregate nodes are below 129 times 300k. At that budget, completed iterations were A=2,384, B=2,322, C=2,811. Nominal depth is not equivalent proof depth: A has recovery reductions, and C can finish with a bound. These numbers measure effort, not strength. On the 80 game-derived positions alone at depth 16, nodes were A=10,158,892, B=14,036,269, C=7,602,291; savings were not confined to artificial bench positions. C accepted bounds in 660/1,280 iterations on that subset.

At depth 16, aspiration attempts were A=7,537, B=7,520, C=5,072. C accepted 1,137 bounds in 2,032 completed iterations (56.0%) and finished 86/129 positions on a bound. Every accepted bound had unsearched root moves; on average only 1.17 of 32.29 root moves were visited in that accepting attempt. Final move differences from A were B=23/129 and C=25/129. At fixed nodes those counts were B=22/129 and C=25/129. At depth 20 they were B=1/16 and C=2/16. Move differences alone do not indicate errors.

Tracing B without changing its search identified 751 ordinary two-high opportunities at depth 16 (restricted to non-large scores). The later attempts consumed **4,411,152 nodes**, 20.0% of B's total. Only 31/751 changed the selected move before iteration completion; however, 157/751 ended below the earlier threshold. This is a local cost measurement on B's trajectory, not an exact counterfactual estimate of C's savings. The larger aggregate C/B reduction includes changed TT/history and predictor trajectories; it is not a direct measurement of useless node work. Some precision work has little immediate decision value; some resolves a real reversal.

Prediction error is absolute difference between the first aspiration center and the eventual in-window result, in internal Value units, excluding initialization and decisive scores. At depth 16 the conditional MAE was A=103.82 (1,898 samples), B=104.53 (1,896), C=116.06 (762). C's bound iterations are censored, so these samples are not directly comparable and do not establish improved prediction. Full counts and other suites are in `results/summary.json`.

## Failure cases and stopping

There were 1,051 completed iterations immediately following a C bound at depth 16. The first attempt failed low in 133. Of 219 that ended exact, **185 ended below the preceding threshold and 74 changed move**. Across all suites, 3,055 accepted bounds were checked to retain exactly beta as score and UCI threshold, with the bound flag set and two consecutive fail-highs.

One reversal is `r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1`: nominal iteration 3 accepted `c3c4` at a lower bound of 2159; iteration 4 began by failing low and completed at 1050 with `f1e1`. These are internal score units, not UCI centipawns. Stopping at the first bound would miss that correction.

The worst cascade was **43 consecutive bound iterations** at 300k nodes in `8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1`. C kept `f5f6`, moving its threshold from -549 to 911 over nominal iterations 3–45; A and B also chose `f5f6` and had much higher eventual evaluations. Thus this cascade demonstrates prediction lag and repeated cheap PV cutoffs, not by itself a wrong move or an inflated final evaluation. It also demonstrates that stopping after two fail-highs per iteration does not bound the number of bound iterations. The longest depth-16 cascade was 15 iterations. Machine-readable reversals and cascades are retained in `results/`.

Within-iteration oscillation is also present in A/B. For `6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1`, B's nominal depth 3 took 21 attempts: depth 3 failed high at 275, depth 2 failed high at 341, then depth 1 failed low at 245. Resetting depth repeated the cycle while widening eventually pushed beta to 5506; it finally returned 97 at depth 1. The move remained `h4f4` throughout. C escaped that precision loop, but then had 14 consecutive bound iterations before finding mate in iteration 16. Ending one kind of cascade can move it across iterations rather than eliminate it. `results/oscillations.json` preserves the attempt logs.

No final verification search is added to the first patch. Returning a legal PV with a flagged lower bound has no structural requirement for an exact score, and ordinary node/time/stop limits remain effective. But these results do **not** prove verification unnecessary for strength. A repeated bounded search can cut off on the same first move without assessing competitors, and starting another search after a hard stop would violate the caller's budget. A future verification must run before soft stopping, honor the existing hard limit, and retain the earlier move/PV/bound if interrupted. A fail-low in a shallower verification must not automatically erase a deeper established result. No such verification/retention heuristic is claimed tested here.

## Correctness and strength

Release and separate instrumented builds reproduced each variant's bench signature exactly. A clean assertion-enabled C build also completed bench with signature 1,189,342. For each variant: three known perft positions passed (startpos depth 5: 4,865,609; Kiwipete depth 4: 4,085,603; rook ending depth 5: 674,624), five node-budget repeat pairs were deterministic, and seven mode scenarios plus infinite/ponder stop checks passed. Actual Syzygy files were unavailable; tablebase preservation is guarded in code but was not exercised with loaded tables. No sanitizer result is claimed.

Fastchess `d6e3a7a71f8437a39836856ae78a3dcdc9bf934c` runs uninstrumented release binaries with one thread, 32 MiB hash, MultiPV=1, concurrency=2 on a host with a two-CPU quota (Intel Xeon Platinum 8573C, three visible CPUs). Compiler: GCC 14.2.0; Python 3.12.14 / python-chess 0.31.4. These are nominal STC/LTC clock settings, **not** CPU-normalized Fishtest runs. Six fixed ten-ply openings are paired with colors swapped at STC, two at LTC. There is no evaluation-based draw or resignation adjudication, since bound scores could confound it; a 200-move draw cap is the only non-rule adjudication.

| Pilot | Candidate vs reference | Games | Wins | Losses | Draws |
|---|---|---:|---:|---:|---:|
| 10+0.1 seconds | B vs A | 12 | 1 | 0 | 11 |
| 10+0.1 seconds | C vs A | 12 | 0 | 0 | 12 |
| 10+0.1 seconds | C vs B | 12 | 1 | 1 | 10 |
| 60+0.6 seconds | C vs B | 4 | 0 | 0 | 4 |

All 40 timed games completed with no crashes, illegal moves, time forfeits, or 200-move cap adjudications. These small, draw-heavy samples cannot establish an Elo gain, equivalence, or safety of final-bound stopping. Fastchess reported C vs B's STC estimate as 0 Elo with roughly +/-82 Elo uncertainty; even that interval is unstable at this sample size. This was not a powered SPRT. The four LTC games were all drawn. Exact commands and logs are checked in. Complete UCI traces, full case lists, and all PGNs are also retained in the delivered `fuzzy-score-results.zip` archive.

## Reproduction

Requires a C++ toolchain, Python with python-chess, the default network, and Fastchess for the optional match pilot. The result directory must be outside the repository. The build helper creates a fresh local clone, leaves the source checkout untouched, and emits separate release/trace binaries and instrumented sources. All variants use the C header layout; A's signature must still match 1,442,480.

```sh
python3 experiments/fuzzy-score/build.py --out /tmp/fuzzy-run --net /path/to/nn-bddd9d57eb87.nnue
export FUZZY_OUT=/tmp/fuzzy-run
python3 experiments/fuzzy-score/correctness.py
python3 experiments/fuzzy-score/run_suite.py
python3 experiments/fuzzy-score/analyze.py
cp experiments/fuzzy-score/openings.epd "$FUZZY_OUT/"
FASTCHESS=/path/to/fastchess python3 experiments/fuzzy-score/run_matches.py
```

The tracer records nominal depth, requested effective depth, alpha, beta, fail-soft value, nodes per attempt, consecutive fail-high count, initial center, root moves visited/available, stop state, move, and completed-iteration score/bound/predictor state. Final decisions and entire UCI PVs are retained. It is not compiled into the engine patch or used for timed matches. Timings from trace runs are not performance measurements.

## Assessment and smallest next experiment

Score fuzziness substantially reduced node work in this corpus. It did not establish that all the saved work was wasted: comparisons were deferred, bounds reversed, and the predictor could remain behind a staircase of bounds. Strength must decide whether the deeper PV/TT work compensates for those omissions. This patch should remain experimental until a properly powered paired test passes.

The smallest next experiment is one change at the accepted bound: raise `averageScore` to at least its established beta threshold, leaving the existing signed-moment update and two-high cutoff unchanged. This tests whether preventing a known lower bound from remaining above the next prediction reduces cascades without feeding the last fail-soft overshoot into the predictor. Keep final verification as a separate subsequent experiment so its cost and benefit can be measured independently.

## Follow-up: recorded LTC move agreement

A separate [complete-game replay](ltc-replay/README.md) compares A/B/C at 10k and 100k nodes per move on 2,423 recorded positions from 16 sampled eight-thread LTC games. It always follows the recorded continuation and preserves per-color search state. Aggregate agreement differences are small and the paired game-bootstrap intervals include zero. The engine patch is unchanged.
