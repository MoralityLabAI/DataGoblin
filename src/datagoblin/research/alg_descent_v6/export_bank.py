"""Fresh v6 banks and independent tuple-key reference at every slot count."""
import json, struct, sys, time
from pathlib import Path
here=Path(__file__).resolve().parent
base=next(p for p in (here.parent/'alg_descent_v5',here.parent/'v5') if (p/'export_bank.py').exists())
sys.path.insert(0,str(base))
import importlib.util
spec=importlib.util.spec_from_file_location('v5_export',base/'export_bank.py')
v5=importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)
CAPS=(36,72,108,252,504,1008)

def main():
    directory=Path(sys.argv[1] if len(sys.argv)>1 else here); directory.mkdir(exist_ok=True,parents=True)
    t=time.perf_counter_ns(); prior,_=v5.training_data(); prior_ns=time.perf_counter_ns()-t
    roots=[]; refs=[]; bank_ns=export_ns=ref_ns=0
    slots=sorted({0,*[c//s for c in CAPS for s in (12,36)]})
    with (directory/'bank.bin').open('wb') as f:
        f.write(b'AD5B'+struct.pack('<I',56))
        for split,base,degs,count in [('iid',170000,(8,12,16,20),24),('ood',180000,(24,32,40,48),32)]:
            for i in range(count):
                t=time.perf_counter_ns(); r=v5.Root(base+i,degs[i%4]); nodes=v5.bank(r,prior); bank_ns+=time.perf_counter_ns()-t
                t=time.perf_counter_ns(); f.write(struct.pack('<IBBH',r.seed,r.n,r.p,len(nodes)))
                for key,actions in nodes:
                    f.write(bytes([len(key),*key,len(actions)]))
                    for a in actions:
                        f.write(bytes([len(a)]))
                        for j,q in a: f.write(struct.pack('<H',j)+bytes([len(q),*q]))
                export_ns+=time.perf_counter_ns()-t
                t=time.perf_counter_ns()
                for s in slots:
                    for policy in ('lru','pending') if s else ('lru',):
                        refs.append(dict(seed=r.seed,slots=s,policy=policy,**v5.replay(r.n,r.p,nodes,s,policy)))
                ref_ns+=time.perf_counter_ns()-t
                roots.append(dict(seed=r.seed,n=r.n,p=r.p,split=split,nodes=len(nodes),unique=len({k for k,_ in nodes})))
    (directory/'reference.json').write_text(json.dumps(dict(roots=roots,refs=refs,prior_ns=prior_ns,bank_ns=bank_ns,export_ns=export_ns,reference_ns=ref_ns),sort_keys=True))
if __name__=='__main__': main()
