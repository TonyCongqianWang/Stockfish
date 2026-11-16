import json,statistics,collections
from pathlib import Path
import os
R=Path(os.environ['FUZZY_OUT']).resolve()
def parse(rec):
    a=[]; it=[]
    for l in rec['lines']:
        if l.startswith('info string ASP '):
            v=l.split()[3:]; a.append(dict(zip(['depth','effective','alpha','beta','value','nodes','failedHigh','center','visited','total','stop'],map(int,v[:11])),move=v[11]))
        elif l.startswith('info string ITER '):
            v=l.split()[3:];it.append(dict(zip(['depth','score','uci','avg','mss','lower','upper','stop','nodes'],map(int,v[:9])),move=v[9]))
    return a,it
allruns={label:json.loads((R/f'suite-{label}.json').read_text()) for label in 'ABC'}
summary={}; failures=[]
for label,runs in allruns.items():
 for kind,limit in [('depth',16),('nodes',300000),('depth',20)]:
    selected=[r for r in runs if (r['kind'],r['limit'])==(kind,limit)]
    if not selected:continue
    s=collections.Counter(); errors=[]; overshoots=[]; maxstreak=0
    for rec in selected:
        aa,ii=parse(rec)
        s['positions']+=1;s['nodes']+=ii[-1]['nodes'] if ii else 0
        s['attempts']+=len(aa);s['completed_attempts']+=sum(not a['stop'] for a in aa)
        s['failhighs']+=sum(a['value']>=a['beta'] and not a['stop'] for a in aa)
        s['faillows']+=sum(a['value']<=a['alpha'] and not a['stop'] for a in aa)
        s['iterations']+=sum(not i['stop'] for i in ii)
        first={};last={}
        for a in aa:first.setdefault(a['depth'],a);last[a['depth']]=a
        for it in ii:
            if it['stop']: continue
            attempts=[a for a in aa if a['depth']==it['depth']]
            for j,a in enumerate(attempts):
                if not a['stop'] and a['failedHigh']>=1 and a['value']>=a['beta'] and abs(a['value'])<750 and abs(a['beta'])<750:
                    s['two_high_opportunities']+=1
                    s['nodes_after_two_highs']+=sum(x['nodes'] for x in attempts[j+1:])
                    s['move_changes_after_two_highs']+=it['move']!=a['move']
                    s['score_below_two_high_threshold']+=not it['lower'] and it['score']<a['beta']
                    break
        prev=None;streak=0
        for it in ii:
            if it['stop']:continue
            d=it['depth']; a=last[d]
            if not it['lower'] and not it['upper'] and abs(it['score'])<31000 and abs(first[d]['center'])<31000:
                errors.append(abs(it['score']-first[d]['center']))
            if prev and prev['lower']:
                s['after_bound']+=1
                low=first[d]['value']<=first[d]['alpha'] and not first[d]['stop']
                s['after_bound_first_faillow']+=low
                if not it['lower'] and not it['upper']:
                    s['bound_followed_exact']+=1
                    violation=it['score']<prev['uci'];s['bound_reversals']+=violation
                    s['bound_next_exact_move_changes']+=it['move']!=prev['move']
                    if violation and label=='C':failures.append(dict(kind=kind,limit=limit,fen=rec['fen'],bound=prev,next=it,first=first[d]))
            if it['lower']:
                s['bound_iterations']+=1
                s['bound_with_unsearched_roots']+=a['visited']<a['total']
                s['bound_roots_visited']+=a['visited'];s['bound_roots_total']+=a['total']
                overshoots.append(a['value']-a['beta']);streak+=1;maxstreak=max(streak,maxstreak)
            else:streak=0
            prev=it
        if ii and ii[-1]['lower']:s['final_lowerbound']+=1
        if ii and ii[-1]['upper']:s['final_upperbound']+=1
        base=next(r for r in allruns['A'] if r['fen']==rec['fen'] and r['kind']==kind and r['limit']==limit)
        s['final_move_changes_vs_A']+=rec['lines'][-1].split()[1]!=base['lines'][-1].split()[1]
    result=dict(s,exact_prediction_mae=statistics.mean(errors) if errors else None,exact_prediction_samples=len(errors),max_bound_streak=maxstreak,max_accepted_overshoot=max(overshoots,default=0))
    summary[f'{label}-{kind}{limit}']=result
(R/'summary.json').write_text(json.dumps(summary,indent=2));(R/'reversals.json').write_text(json.dumps(failures,indent=2)); print(json.dumps(summary,indent=2))
