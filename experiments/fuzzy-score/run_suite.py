import subprocess, json, re, time, concurrent.futures
from pathlib import Path
import chess
import os
R=Path(os.environ['FUZZY_OUT']).resolve(); repo=Path(__file__).resolve().parents[2]
s=(repo/'src/benchmark.cpp').read_text()
section=s[s.index('const std::vector<std::string> Defaults'):s.index('// clang-format on')]
fens=[]; is960=False
for line in section.splitlines():
    m=re.search(r'"([^"]+)"',line)
    if not m: continue
    fen=m[1]
    if fen.startswith('setoption'): is960='true' in fen; continue
    if is960: continue
    fen=fen.split(' moves ')[0]
    if len(fen.split())==4: fen+=' 0 1'
    fens.append(fen)
section=s[s.index('const std::vector<std::vector<std::string>> BenchmarkPositions'):]
gamefens=re.findall(r'"([^"\n]+ [wb] [^"\n]+)"',section)
for fen in gamefens[::3][:80]:
    try: chess.Board(fen)
    except ValueError: continue
    if fen not in fens: fens.append(fen)
(R/'positions.json').write_text(json.dumps(fens,indent=2))
def read_until(p,token):
    lines=[]
    while True:
        line=p.stdout.readline()
        if not line: raise RuntimeError('Engine exited: '+str(p.poll()))
        lines.append(line.strip())
        if line.startswith(token): return lines

def run(label):
    p=subprocess.Popen([str(R/(label+'-trace'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    def send(s): p.stdin.write(s+'\n'); p.stdin.flush()
    send('uci'); read_until(p,'uciok')
    send('setoption name Hash value 32'); send('setoption name Threads value 1'); send('isready'); read_until(p,'readyok')
    data=[]
    for kind,limit,positions in [('depth',16,fens),('nodes',300000,fens),('depth',20,fens[2:18])]:
        for i,fen in enumerate(positions):
            send('ucinewgame'); send('isready'); read_until(p,'readyok')
            send('position fen '+fen); send(f'go {kind} {limit}')
            start=time.monotonic(); lines=read_until(p,'bestmove')
            board=chess.Board(fen); best=lines[-1].split()[1]
            assert best=='(none)' or chess.Move.from_uci(best) in board.legal_moves, (label,fen,best)
            for line in lines:
                if line.startswith('info depth') and ' pv ' in line:
                    b=board.copy()
                    for uci in line.split(' pv ')[1].split():
                        move=chess.Move.from_uci(uci); assert move in b.legal_moves,(label,fen,line,uci); b.push(move)
            data.append(dict(fen=fen,kind=kind,limit=limit,seconds=time.monotonic()-start,lines=lines))
            (R/f'suite-{label}.json').write_text(json.dumps(data))
        print(label,kind,limit,len(positions),'done',flush=True)
    send('quit'); p.wait(timeout=10)
    return label,len(data)
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        for result in ex.map(run,['A','B','C']): print(result,flush=True)
