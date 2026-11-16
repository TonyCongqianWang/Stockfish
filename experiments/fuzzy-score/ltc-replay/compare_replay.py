#!/usr/bin/env python3
"""Summarize exact recorded-move agreement; paired bootstrap resamples whole games."""
import argparse,collections,json,random,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
data={}
for f in sorted(a.results.glob('[ABC]-*.jsonl')):
 label,n=f.stem.split('-');data[label,int(n)]=[json.loads(l) for l in f.read_text().splitlines()]
base=next(iter(data.values()));keys=[(r['game'],r['ply'],r['fen'],r['played']) for r in base]
for rows in data.values():assert [(r['game'],r['ply'],r['fen'],r['played']) for r in rows]==keys
filters={'all':lambda r:True,'nonforced':lambda r:r['legal_moves']>1,'first_40_plies_nonforced':lambda r:r['ply']<40 and r['legal_moves']>1,'quiet_nonforced':lambda r:r['legal_moves']>1 and not r['capture'] and not r['promotion'] and not r['in_check'],'17plus_pieces_nonforced':lambda r:r['legal_moves']>1 and r['pieces']>=17,'9to16_pieces_nonforced':lambda r:r['legal_moves']>1 and 9<=r['pieces']<=16,'8orless_pieces_nonforced':lambda r:r['legal_moves']>1 and r['pieces']<=8,'teacher_new_nonforced':lambda r:r['legal_moves']>1 and r['teacher'].startswith('New-'),'teacher_base_nonforced':lambda r:r['legal_moves']>1 and r['teacher'].startswith('Base-')}
summary={};pairs={}
for (label,n),rows in sorted(data.items()):
 cells={}
 for name,predicate in filters.items():
  selected=[r for r in rows if predicate(r)];count=len(selected);matched=sum(r['match'] for r in selected)
  cells[name]=dict(positions=count,matches=matched,percent=100*matched/count if count else None,lower_bounds=sum(r['lower'] for r in selected),upper_bounds=sum(r['upper'] for r in selected))
  bygame=collections.defaultdict(list)
  for r in selected:bygame[r['game']].append(r['match'])
  cells[name]['game_mean_percent']=statistics.mean(100*sum(v)/len(v) for v in bygame.values()) if bygame else None
 cells['nodes_reported']=sum(r['nodes'] or 0 for r in rows);cells['median_reported_depth']=statistics.median(r['depth'] for r in rows if r['depth'] is not None)
 summary[f'{label}-{n}']=cells
for n in sorted({n for _,n in data}):
 for arm,ref in [('B','A'),('C','A'),('C','B')]:
  for name in ['all','nonforced','first_40_plies_nonforced','quiet_nonforced']:
   counts=collections.defaultdict(lambda:[0,0]);better=worse=changed=0
   for x,y in zip(data[arm,n],data[ref,n]):
    assert filters[name](x)==filters[name](y)
    if not filters[name](x):continue
    counts[x['game']][0]+=int(x['match'])-int(y['match']);counts[x['game']][1]+=1
    better+=x['match'] and not y['match'];worse+=y['match'] and not x['match'];changed+=x['chosen']!=y['chosen']
   groups=list(counts.values());rng=random.Random(20261116);boot=[]
   for _ in range(10000):
    draw=[groups[rng.randrange(len(groups))] for _ in groups];den=sum(x[1] for x in draw)
    if den:boot.append(100*sum(x[0] for x in draw)/den)
   boot.sort();pairs[f'{arm}-vs-{ref}-{n}-{name}']=dict(delta_percentage_points=100*sum(x[0] for x in groups)/sum(x[1] for x in groups),bootstrap_95_interval=[boot[int(.025*len(boot))],boot[int(.975*len(boot))]],candidate_only_matches=better,reference_only_matches=worse,changed_moves=changed,games=len(groups))
result=dict(positions=len(base),games=len({r['game'] for r in base}),searches=sum(map(len,data.values())),rates=summary,paired_differences=pairs)
a.out.write_text(json.dumps(result,indent=2))
for k,v in summary.items():print(k,'all',v['all'],'first40',v['first_40_plies_nonforced'])
for k,v in pairs.items():
 if k.startswith('C-') and (k.endswith('-all') or k.endswith('-first_40_plies_nonforced')):print(k,v)
