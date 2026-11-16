#!/usr/bin/env python3
"""Reservoir-sample complete PGNs, without parsing the entire source with python-chess."""
import argparse,gzip,hashlib,io,json,random
from pathlib import Path
import chess.pgn
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--games',type=int,default=16);p.add_argument('--seed',type=int,default=20261116);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def records(f):
    lines=[]
    for line in f:
        if line.startswith('[Event ') and lines:
            yield ''.join(lines);lines=[]
        lines.append(line)
    if lines:yield ''.join(lines)
rng=random.Random(a.seed);sample=[];count=0
with gzip.open(a.source,'rt',encoding='utf-8-sig') as f:
    for idx,s in enumerate(records(f)):
        count=idx+1
        if len(sample)<a.games:sample.append((idx,s))
        else:
            j=rng.randrange(count)
            if j<a.games:sample[j]=(idx,s)
manifest=[];texts=[];positions=0
for idx,s in sorted(sample):
    g=chess.pgn.read_game(io.StringIO(s));assert g is not None and not g.errors
    b=g.board();moves=list(g.mainline_moves());assert moves
    for m in moves:assert m in b.legal_moves;b.push(m)
    positions+=len(moves);texts.append(s.strip())
    manifest.append(dict(source_index=idx,headers=dict(g.headers),moves=len(moves)))
(a.out/'sample.pgn').write_text('\n\n'.join(texts)+'\n')
meta=dict(source_url='https://tests.stockfishchess.org/api/run_pgns/6af233258e7fb2e78bc1cc0b.pgn.gz',source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),source_games=count,seed=a.seed,games=len(manifest),positions=positions,selection='uniform reservoir sampling of complete games in archive order',manifest=manifest)
(a.out/'sample-manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='manifest'},indent=2))
