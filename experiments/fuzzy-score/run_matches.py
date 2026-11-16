from pathlib import Path
import subprocess,json,time,os
R=Path(os.environ['FUZZY_OUT']).resolve()
fast=os.environ.get('FASTCHESS','fastchess')
for label,a,b,tc,rounds in [('STC-B-A','B','A','10+0.1',6),('STC-C-A','C','A','10+0.1',6),('STC-C-B','C','B','10+0.1',6),('LTC-C-B','C','B','60+0.6',2)]:
    args=[fast,'-engine','cmd='+str(R/(a+'-release')),'name='+a,'-engine','cmd='+str(R/(b+'-release')),'name='+b,'-each','tc='+tc,'option.Threads=1','option.Hash=32','option.MultiPV=1','-openings','file='+str(R/'openings.epd'),'format=epd','order=sequential','-rounds',str(rounds),'-repeat','-concurrency','2','-maxmoves','200','-pgnout','file='+str(R/(label+'.pgn')),'nodes=true','seldepth=true','timeleft=true','append=false','-autosaveinterval','0','-ratinginterval','2','-scoreinterval','2']
    (R/(label+'-command.json')).write_text(json.dumps(args,indent=2))
    print('Starting',label,flush=True)
    with (R/(label+'.log')).open('w') as f:
        start=time.monotonic();subprocess.run(args,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Finished',label,round(time.monotonic()-start,1),'seconds',flush=True)
