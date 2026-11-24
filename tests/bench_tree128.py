"""Alternate matched PGO layouts; report medians and exact bench signatures.

All three routing layouts must search identical node counts with the same net.
The 192-stack control requires its own matching network. Random validation nets
measure implementation cost only; repeat with independently trained matched nets.
"""
import argparse
import json
from pathlib import Path
import random
import re
import statistics
import subprocess

ap=argparse.ArgumentParser()
ap.add_argument('--build-root',type=Path,required=True)
ap.add_argument('--tree-net',type=Path,required=True)
ap.add_argument('--control-net',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--repeats',type=int,default=5)
ap.add_argument('--depth',type=int,default=13)
args=ap.parse_args()
rng=random.Random(128);results={name:[] for name in ['binary','small_switch','mixed','control']}
for repeat in range(args.repeats):
    order=list(results);rng.shuffle(order)
    for name in order:
        net=args.control_net if name=='control' else args.tree_net
        command=f'setoption name EvalFile value {net.resolve()}\nbench 16 1 {args.depth} default depth\nquit\n'
        r=subprocess.run([str((args.build_root/name/'src/stockfish').resolve())],input=command,text=True,capture_output=True,check=True,timeout=120)
        all_output=r.stdout+r.stderr
        def metric(label):
            return int(re.search(label+r'\s*:\s*(\d+)',all_output).group(1))
        row=dict(nps=metric('Nodes/second'),nodes=metric('Nodes searched'),time_ms=metric(r'Total time \(ms\)'))
        results[name].append(row)
        print(repeat,name,row,flush=True)
for repeat in range(args.repeats):
    assert len({results[name][repeat]['nodes'] for name in ('binary','small_switch','mixed')})==1
summary={name:dict(median_nps=statistics.median(r['nps'] for r in rows),min_nps=min(r['nps'] for r in rows),max_nps=max(r['nps'] for r in rows)) for name,rows in results.items()}
report=dict(depth=args.depth,repeats=args.repeats,results=results,summary=summary,note='PGO profiles used each matching validation network. The tree layouts have identical outputs/search signatures. Randomly initialized networks; control search signature differs; no strength inference.')
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(summary,flush=True)
