import subprocess, json, sys
from pathlib import Path
import chess
import os
R=Path(os.environ['FUZZY_OUT']).resolve()
from run_suite import read_until
results=[]
for label in ['A','B','C']:
    exe=str(R/(label+'-release'))
    for pos,depth,count in [('startpos',5,4865609),('fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',4,4085603),('fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',5,674624)]:
        s=subprocess.check_output([exe],input=f'position {pos}\ngo perft {depth}\nquit\n',text=True)
        assert f'Nodes searched: {count}' in s
    p=subprocess.Popen([exe],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def send(s):p.stdin.write(s+'\n');p.stdin.flush()
    send('uci');read_until(p,'uciok')
    trials=[]
    for n in [100,1000,10000,100000,300000]:
        repeats=[]
        for _ in range(2):
            send('ucinewgame');send('isready');read_until(p,'readyok')
            send('position startpos');send(f'go nodes {n}')
            lines=read_until(p,'bestmove'); repeats.append([x for x in lines if x.startswith('info depth') or x.startswith('bestmove')])
        # Time/nps vary, so compare deterministic fields only.
        import re
        normalize=lambda lines:[re.sub(r' (nps|time) \d+','',x) for x in lines]
        assert normalize(repeats[0])==normalize(repeats[1]),(label,n)
    for options,pos,go in [
        (['MultiPV value 4'],'startpos','depth 12'),
        (['MultiPV value 1','Threads value 2'],'startpos','depth 12'),
        (['Threads value 1','Skill Level value 5'],'startpos','depth 8'),
        (['Skill Level value 20','UCI_Chess960 value true'],'fen nqbnrkrb/pppppppp/8/8/8/8/PPPPPPPP/NQBNRKRB w KQkq - 0 1','depth 10'),
        (['UCI_Chess960 value false'],'fen 7k/8/5KQ1/8/8/8/8/8 w - - 0 1','mate 2'),
        ([], 'startpos','wtime 1000 btime 1000 winc 10 binc 10'),
        ([], 'startpos','movetime 50'),
    ]:
        for opt in options:send('setoption name '+opt)
        send('ucinewgame');send('isready');read_until(p,'readyok');send('position '+pos);send('go '+go)
        lines=read_until(p,'bestmove'); trials.append(dict(options=options,pos=pos,go=go,lines=lines))
        b=chess.Board() if pos=='startpos' else chess.Board(pos[4:],chess960='nqbn' in pos)
        move=chess.Move.from_uci(lines[-1].split()[1]); assert move in b.legal_moves
    for cmd in ['go infinite','go ponder wtime 500 btime 500']:
        send('ucinewgame');send('position startpos');send(cmd)
        read_until(p,'info depth 8')
        send('stop'); lines=read_until(p,'bestmove'); assert len(lines[-1].split())>=2
    send('quit');assert p.wait(timeout=10)==0
    results.append(dict(variant=label,perft=3,reproducibility=5,scenarios=trials))
(R/'correctness.json').write_text(json.dumps(results,indent=2)); print('PASS: all variants: 3 perft, 5 repeat pairs, 7 mode checks, infinite/ponder stop')
