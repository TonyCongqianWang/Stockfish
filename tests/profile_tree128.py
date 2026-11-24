"""Build isolated PGO layouts with a matching external network.

python tests/profile_tree128.py --tree-net tree128.nnue --control-net control.nnue --build-root /path/to/new/builds
Uses normal Stockfish profile-build/bench, CPPFLAGS (preserved by recursive PGO
make targets), and separate source/object/profile directories for every layout.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--tree-net',type=Path,required=True)
ap.add_argument('--control-net',type=Path,required=True)
ap.add_argument('--build-root',type=Path,required=True)
ap.add_argument('--jobs',type=int,default=4)
args=ap.parse_args()
source=Path(__file__).resolve().parents[1]
args.build_root.mkdir(parents=True,exist_ok=True)
for name,layout in [('binary',0),('small_switch',1),('mixed',2),('control',0)]:
    dest=args.build_root/name
    if dest.exists():
        raise SystemExit(f'Refusing to overwrite existing build: {dest}')
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('.git','*.o','*.nnue','stockfish','*.gcda','*.gcno','.depend','__pycache__'))
    net=(args.control_net if name=='control' else args.tree_net).resolve()
    script=dest/'profile.sh'
    script.write_text('#!/bin/sh\nprintf "setoption name EvalFile value '+str(net)+'\\nbench\\nquit\\n" | ./stockfish\n')
    flags=f'-DNNUE_EMBEDDING_OFF -DTREE128_LAYOUT={layout}'+(' -DTREE128_CONTROL' if name=='control' else '')
    with (args.build_root/f'{name}.log').open('w') as log:
        subprocess.run(['make',f'-j{args.jobs}','-o','net','profile-build','ARCH=x86-64-avx2',f'CPPFLAGS={flags}',f'PGOBENCH=sh {script}'],cwd=dest/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Built PGO {name}',flush=True)
