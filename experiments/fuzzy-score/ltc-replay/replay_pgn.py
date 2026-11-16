#!/usr/bin/env python3
"""Search every recorded position, then push the PGN move rather than the search choice."""
import argparse,concurrent.futures,json,re,time,subprocess
from pathlib import Path
import chess,chess.pgn

def load(path):
    games=[]
    with path.open() as f:
        while (g:=chess.pgn.read_game(f)) is not None:
            assert not g.errors;games.append(g)
    return games

class Engine:
    def __init__(self,path):
        self.p=subprocess.Popen([str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        self.send('uci');self.until('uciok')
        for option in ['Threads value 1','Hash value 32','MultiPV value 1']:self.send('setoption name '+option)
        self.send('isready');self.until('readyok')
    def send(self,line):self.p.stdin.write(line+'\n');self.p.stdin.flush()
    def until(self,prefix):
        lines=[]
        while True:
            line=self.p.stdout.readline()
            if not line:raise RuntimeError('Engine exited: '+str(self.p.poll())+' '+self.p.stderr.read())
            lines.append(line.strip())
            if line.startswith(prefix):return lines
    def newgame(self):self.send('ucinewgame');self.send('isready');self.until('readyok')
    def play(self,root,prefix,budget):
        self.send('position fen '+root+(' moves '+' '.join(prefix) if prefix else ''))
        self.send('go nodes '+str(budget));lines=self.until('bestmove')
        last=next((line for line in reversed(lines) if line.startswith('info depth ') and ' score ' in line),'')
        tokens=last.split()
        def number(key):return int(tokens[tokens.index(key)+1]) if key in tokens else None
        pv=tokens[tokens.index('pv')+1:] if 'pv' in tokens else []
        return chess.Move.from_uci(lines[-1].split()[1]),dict(nodes=number('nodes'),depth=number('depth'),seldepth=number('seldepth'),lower='lowerbound' in tokens,upper='upperbound' in tokens,score=' '.join(tokens[tokens.index('score')+1:tokens.index('score')+3]) if 'score' in tokens else None,pv=pv)
    def quit(self):self.send('quit');self.p.wait(timeout=10)

def replay(label,budget,args):
    games=load(args.pgn);engines=[Engine(args.bin_dir/(label+'-release')) for _ in range(2)]
    rows=[];path=args.out/f'{label}-{budget}.jsonl'
    try:
        with path.open('w') as f:
            for gid,g in enumerate(games):
                board=g.board();root=board.fen();prefix=[];start=time.monotonic()
                for e in engines:e.newgame()
                for ply,node in enumerate(g.mainline()):
                    played=node.move;assert played in board.legal_moves
                    teacher=g.headers['White' if board.turn else 'Black'];fen=board.fen();legal_count=board.legal_moves.count()
                    pre=dict(game=gid,ply=ply,fen=fen,played=played.uci(),teacher=teacher,legal_moves=legal_count,pieces=len(board.piece_map()),in_check=board.is_check(),capture=board.is_capture(played),promotion=bool(played.promotion),teacher_comment=node.comment)
                    t=time.monotonic()
                    # Separate per-color engines preserve each side's own history/TT.
                    chosen,info=engines[int(board.turn)].play(root,prefix,budget)
                    assert chosen in board.legal_moves
                    row=dict(pre,variant=label,budget=budget,chosen=chosen.uci(),match=chosen==played,seconds=time.monotonic()-t,nodes=info.get('nodes'),depth=info.get('depth'),seldepth=info.get('seldepth'),lower=info['lower'],upper=info['upper'],score=str(info.get('score')),pv=info['pv'])
                    f.write(json.dumps(row,separators=(',',':'))+'\n');rows.append(row)
                    board.push(played);prefix.append(played.uci())
                    assert board.fen()==node.board().fen(),(gid,ply)
                f.flush();print(label,budget,'game',gid+1,'/',len(games),'plies',ply+1,'seconds',round(time.monotonic()-start,2),flush=True)
    finally:
        for e in engines:e.quit()
    return dict(variant=label,budget=budget,positions=len(rows),matches=sum(r['match'] for r in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pgn',type=Path,required=True);p.add_argument('--bin-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--budgets',type=int,nargs='+',default=[10000,100000]);p.add_argument('--variants',nargs='+',default=list('ABC'));p.add_argument('--jobs',type=int,default=2);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        futures=[ex.submit(replay,label,n,a) for n in a.budgets for label in a.variants]
        summary=[f.result() for f in futures]
    (a.out/'replay-completed.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
