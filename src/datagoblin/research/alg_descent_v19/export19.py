"""Recreate the prospective V19 public graph bank from the V11 generator."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'prior' if (HERE/'prior').is_dir() else HERE.parent/'alg_descent_v11'))
from proof_planning import Graph
with (HERE/'bank19.txt').open('w') as f:
    print(36,file=f)
    for split,base,count,ns,fs in [('iid',1410000,12,(10,11,12,13),('uniform','clustered')),('ood',1420000,24,(16,17,18),('shared','uniform','clustered'))]:
        for i in range(count):
            g=Graph(base+i,ns[i%len(ns)],fs[(i+i//len(ns))%len(fs)] if split=='ood' else fs[(i//len(ns)+i)%2])
            print(g.seed,g.n,len(g.rules),file=f)
            for r in g.rules:print(r.head,r.mask,r.cost,file=f)
