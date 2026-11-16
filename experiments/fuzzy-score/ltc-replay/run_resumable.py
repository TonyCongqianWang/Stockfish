#!/usr/bin/env python3
"""Checkpointed whole-game replay; game scheduling does not change search state."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path

from replay_pgn import Engine, load


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def one_game(engines, game, gid, label, budget, stop):
    board = game.board()
    root = board.fen()
    prefix, rows = [], []
    for engine in engines:
        engine.newgame()
    for ply, node in enumerate(game.mainline()):
        if stop.is_set():
            return None
        played = node.move
        assert played in board.legal_moves
        pre = dict(game=gid, ply=ply, fen=board.fen(), played=played.uci(),
                   teacher=game.headers['White' if board.turn else 'Black'],
                   legal_moves=board.legal_moves.count(), pieces=len(board.piece_map()),
                   in_check=board.is_check(), capture=board.is_capture(played),
                   promotion=bool(played.promotion), teacher_comment=node.comment)
        start = time.monotonic()
        chosen, info = engines[int(board.turn)].play(root, prefix, budget)
        assert chosen in board.legal_moves
        rows.append(dict(pre, variant=label, budget=budget, chosen=chosen.uci(),
                         match=chosen == played, seconds=time.monotonic() - start,
                         nodes=info.get('nodes'), depth=info.get('depth'),
                         seldepth=info.get('seldepth'), lower=info['lower'],
                         upper=info['upper'], score=str(info.get('score')), pv=info['pv']))
        board.push(played)
        prefix.append(played.uci())
        assert board.fen() == node.board().fen(), (gid, ply)
    return rows


def report(out, config, manifest, summary):
    rates, pairs = summary['rates'], summary['paired_differences']
    n, games = summary['positions'], summary['games']
    text = [f'# Recorded LTC move agreement: {games} games', '',
            f'Complete: {summary["searches"]:,} searches over {n:,} recorded positions.', '',
            'A is original master; B removes failHighRecovery only; C is the unchanged '
            'two-fail-high fuzzy patch. Each search uses one thread, MultiPV=1 and Hash=32 MiB. '
            'Each color retains its own TT/history. Every continuation follows the recorded '
            'move, including after a candidate disagrees.', '',
            'The sample is a new uniform reservoir sample, seed 20261116, from the same '
            '59,974-game eight-thread LTC archive. It is not a pool of the earlier 16-game '
            'results. The teacher engines and networks differ from the student builds. '
            'PGNs begin at their recorded post-book FEN.', '',
            '| Nodes/move | Original A | Removal-only B | Fuzzy C | C−A (pp) | Game-bootstrap 95% interval |',
            '|---|---:|---:|---:|---:|---:|']
    for budget in config['budgets']:
        values = [rates[f'{label}-{budget}']['all']['percent'] for label in 'ABC']
        diff = pairs[f'C-vs-A-{budget}-all']
        lo, hi = diff['bootstrap_95_interval']
        text.append(f'| {budget:,} | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% '
                    f'| {diff["delta_percentage_points"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    for subset, title in [('nonforced', 'Positions with more than one legal move'),
                          ('first_40_plies_nonforced', 'First 40 recorded plies, nonforced'),
                          ('quiet_nonforced', 'Quiet positions, nonforced')]:
        count = rates[f'A-{config["budgets"][0]}'][subset]['positions']
        text += ['', f'## {title} ({count:,} positions)', '',
                 '| Nodes/move | A | B | C | C−B (pp) | Game-bootstrap 95% interval |',
                 '|---|---:|---:|---:|---:|---:|']
        for budget in config['budgets']:
            values = [rates[f'{label}-{budget}'][subset]['percent'] for label in 'ABC']
            diff = pairs[f'C-vs-B-{budget}-{subset}']
            lo, hi = diff['bootstrap_95_interval']
            text.append(f'| {budget:,} | {values[0]:.3f}% | {values[1]:.3f}% | {values[2]:.3f}% '
                        f'| {diff["delta_percentage_points"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    text += ['', 'Intervals use 10,000 paired bootstrap resamples of whole games, not independent '
             'moves. The summary also includes game-weighted rates, piece-count and teacher splits, '
             'exclusive matches, changed choices, and reported bound counts. Exact move '
             'disagreements may be harmless alternatives; this proxy does not establish an Elo '
             'difference and does not exercise clock-based time management.', '',
             'Inputs and binary SHA-256 hashes are frozen in config.json. All candidate choices '
             'were checked for legality, and every advanced board was checked against the PGN. '
             'The analysis checks identical position/label sequences across all six runs. '
             'Complete per-position records and the sample PGNs are in the accompanying archive.', '']
    (out / 'REPORT.md').write_text('\n'.join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[10000, 100000])
    parser.add_argument('--variants', nargs='+', default=list('ABC'))
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--game-indices', type=int, nargs='+')
    parser.add_argument('--replay-only', action='store_true')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_file = (args.out / 'run.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    games = load(args.pgn)
    selected = args.game_indices if args.game_indices is not None else list(range(len(games)))
    lengths = {gid: len(list(games[gid].mainline_moves())) for gid in selected}
    cfg = dict(sample_sha256=digest(args.pgn), games=len(games), selected_games=selected,
               budgets=args.budgets, variants=args.variants,
               binary_sha256={label: digest(args.bin_dir / (label + '-release')) for label in args.variants},
               threads=1, multipv=1, hash_mib=32, engine_commit='d8b7027952c15ca02c15b40110f44ffdafd4e0cb')
    config_path = args.out / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == cfg, 'Inputs changed; use a fresh output directory'
    else:
        atomic_json(config_path, cfg)
    snapshots = args.out / 'scripts'
    snapshots.mkdir(exist_ok=True)
    for name in ['run_resumable.py', 'replay_pgn.py', 'compare_replay.py', 'sample_pgn.py']:
        target = snapshots / name
        if not target.exists():
            shutil.copyfile(Path(__file__).with_name(name), target)
    if not (args.out / 'sample.pgn').exists():
        shutil.copyfile(args.pgn, args.out / 'sample.pgn')
    work = queue.Queue()
    completed = []
    total_weight = 0
    for budget in args.budgets:
        for label in args.variants:
            directory = args.out / 'games' / f'{label}-{budget}'
            directory.mkdir(parents=True, exist_ok=True)
            for gid in selected:
                path = directory / f'{gid:04d}.jsonl'
                meta_path = path.with_suffix('.summary.json')
                total_weight += lengths[gid] * budget
                if path.exists() and meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    assert meta['positions'] == lengths[gid] and meta['sha256'] == digest(path)
                    completed.append(meta)
                else:
                    work.put((label, budget, gid, path))
    total_jobs = len(selected) * len(args.budgets) * len(args.variants)
    stop, done = threading.Event(), threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stop.set())
    mutex = threading.Lock()
    start = time.monotonic()
    prior_elapsed = 0
    progress_path = args.out / 'progress.json'
    if progress_path.exists():
        prior_elapsed = json.loads(progress_path.read_text()).get('elapsed_seconds', 0)
    active = {}
    stage = 'running'

    def progress():
        elapsed = prior_elapsed + time.monotonic() - start
        weight = sum(meta['positions'] * meta['budget'] for meta in completed)
        value = dict(status=stage, pid=os.getpid(), updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                     sample_games=len(selected), completed_game_runs=len(completed), total_game_runs=total_jobs,
                     completed_searches=sum(meta['positions'] for meta in completed),
                     total_searches=sum(lengths.values()) * len(args.budgets) * len(args.variants),
                     node_budget_weight_fraction=weight / total_weight, elapsed_seconds=elapsed,
                     estimated_remaining_seconds=elapsed * (total_weight - weight) / weight if weight else None,
                     active=active.copy(), report=str(args.out / 'REPORT.md'),
                     archive=str(args.out / 'ltc-replay-500-results.zip'))
        atomic_json(progress_path, value)

    def heartbeat():
        while not done.wait(10):
            with mutex:
                progress()

    def worker(worker_id):
        engines, current_label = [], None
        try:
            while not stop.is_set():
                try:
                    label, budget, gid, path = work.get_nowait()
                except queue.Empty:
                    break
                if label != current_label:
                    for engine in engines:
                        engine.quit()
                    engines = [Engine(args.bin_dir / (label + '-release')) for _ in range(2)]
                    current_label = label
                with mutex:
                    active[str(worker_id)] = dict(variant=label, budget=budget, game=gid)
                    progress()
                game_start = time.monotonic()
                rows = one_game(engines, games[gid], gid, label, budget, stop)
                if rows is None:
                    break
                assert len(rows) == lengths[gid]
                tmp = path.with_suffix('.tmp')
                with tmp.open('w') as f:
                    for row in rows:
                        f.write(json.dumps(row, separators=(',', ':')) + '\n')
                tmp.replace(path)
                meta = dict(variant=label, budget=budget, game=gid, positions=len(rows),
                            matches=sum(row['match'] for row in rows), sha256=digest(path),
                            seconds=time.monotonic() - game_start)
                atomic_json(path.with_suffix('.summary.json'), meta)
                with mutex:
                    completed.append(meta)
                    active.pop(str(worker_id), None)
                    progress()
                    print(time.strftime('%H:%M:%S UTC', time.gmtime()), label, budget, 'game', gid,
                          'completed', len(completed), '/', total_jobs, flush=True)
        except BaseException:
            stop.set()
            raise
        finally:
            for engine in engines:
                engine.quit()
            with mutex:
                active.pop(str(worker_id), None)

    with mutex:
        progress()
    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(worker, wid) for wid in range(args.jobs)]
            for future in futures:
                future.result()
        if stop.is_set():
            stage = 'paused'
            return
        assert len(completed) == total_jobs
        if args.replay_only:
            stage = 'completed'
            return
        assert args.variants == list('ABC') and args.game_indices is None
        stage = 'assembling'
        with mutex:
            progress()
        results = args.out / 'results'
        results.mkdir(exist_ok=True)
        for budget in args.budgets:
            for label in args.variants:
                with (results / f'{label}-{budget}.jsonl').open('wb') as f:
                    for gid in selected:
                        with (args.out / 'games' / f'{label}-{budget}' / f'{gid:04d}.jsonl').open('rb') as g:
                            shutil.copyfileobj(g, f)
        stage = 'analyzing'
        with mutex:
            progress()
        with (args.out / 'analysis.log').open('w') as f:
            subprocess.run([sys.executable, str(snapshots / 'compare_replay.py'), '--results', str(results),
                            '--out', str(args.out / 'summary.json')], stdout=f, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((args.out / 'summary.json').read_text())
        report(args.out, cfg, None, summary)
        atomic_json(args.out / 'completed-game-runs.json', completed)
        stage = 'packaging'
        with mutex:
            progress()
        archive = args.out / 'ltc-replay-500-results.zip'
        tmp_archive = archive.with_suffix('.tmp')
        with zipfile.ZipFile(tmp_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in results.glob('*.jsonl'):
                z.write(path, 'positions/' + path.name)
            for name in ['sample.pgn', 'sample-manifest.json', 'config.json', 'summary.json',
                         'REPORT.md', 'analysis.log', 'completed-game-runs.json', 'source-run.json']:
                path = args.out / name
                if path.exists():
                    z.write(path, name)
            for path in snapshots.glob('*.py'):
                z.write(path, 'scripts/' + path.name)
        with zipfile.ZipFile(tmp_archive) as z:
            assert z.testzip() is None
        tmp_archive.replace(archive)
        stage = 'completed'
    except BaseException:
        stage = 'failed'
        (args.out / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        done.set()
        ticker.join()
        with mutex:
            progress()
        lock_file.close()


if __name__ == '__main__':
    main()
