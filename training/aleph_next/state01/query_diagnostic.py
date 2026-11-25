"""Post-hoc exact-query and grammar-misspecification diagnostics for STATE-06."""
import json,time
from pathlib import Path
import numpy as np
from state06 import library,splits,panels,BUDGETS,N,Q,CLASSES

def entropy(p,axis=-1):
    p=np.asarray(p,dtype=float);return -np.sum(p*np.log2(np.maximum(p,1e-300)),axis=axis)

def acquisition(f,cl,prior,m,policy,used):
    # Exact conditional predictive distribution p(C,Y_x | observations).
    w=prior[m];w=w/w.sum();hist=np.zeros((N,4,Q));idx=np.arange(N)[None,:]
    np.add.at(hist,(np.broadcast_to(idx,(len(m),N)),cl[m,None],f[m]),w[:,None])
    py=hist.sum(1); scores=entropy(py)
    if policy=='class':
        pc=hist.sum(2); scores+=entropy(pc)-entropy(hist.reshape(N,-1))
    scores[list(used)]=-np.inf
    return int(scores.argmax())

def counts(f,cl,prior,m,truth,observed):
    mass=np.bincount(cl[m],weights=prior[m],minlength=4)
    agree=np.all(f[m]==f[m[:1]],axis=0)
    unseen=agree.copy();unseen[list(observed)]=False
    return np.array([int(mass.argmax()==cl[truth]),int(np.count_nonzero(mass)==1),int(len(m)==1),int(agree.sum()),int(unseen.sum())])

def run():
    f,cl,_,ood=library();g,i,o=splits(cl,ood);p=panels(f,cl,i,o);prior=1/np.bincount(cl,minlength=4)[cl];all_ids=np.arange(len(f));seen=np.flatnonzero(~ood)
    out={'restricted_grammar':{},'active':{}};start=time.process_time()
    for key,(ids,ix,_,_) in p.items():
        miss=unique_wrong=false_coord=bad_rows=accepted=0
        for j,cols in zip(ids,ix):
            m=seen
            for x in cols:m=m[f[m,x]==f[j,x]]
            if not len(m):miss+=1;continue
            accepted+=1;unique_wrong+=int(len(m)==1 and m[0]!=j)
            mask=np.all(f[m]==f[m[:1]],axis=0)&(f[m[0]]!=f[j]);false_coord+=int(mask.sum());bad_rows+=int(mask.any())
        out['restricted_grammar'][key]=dict(n=len(ids),accepted=accepted,rejected=miss,false_unique_function=unique_wrong,false_certified_coordinates=false_coord,cases_with_false_certificate=bad_rows)
    for policy in ('function','class'):
        for group in ('iid','template_ood'):
            ids,_,_,_=p[f'{group}/16']; acc={b:np.zeros(5,dtype=int) for b in BUDGETS}; first=acquisition(f,cl,prior,all_ids,policy,[])
            for j in ids:
                m=all_ids;used=[]
                for b in range(1,17):
                    if b==1:x=first
                    elif len(m)==1:x=next(k for k in range(N) if k not in used)
                    else:x=acquisition(f,cl,prior,m,policy,used)
                    used.append(x);m=m[f[m,x]==f[j,x]];assert j in m
                    if b in acc:acc[b]+=counts(f,cl,prior,m,j,used)
            out['active'][f'{policy}/{group}']={str(b):dict(n=len(ids),correct=int(v[0]),unique_class=int(v[1]),unique_function=int(v[2]),certified_coordinates=int(v[3]),certified_unqueried=int(v[4])) for b,v in acc.items()};print(policy,group,out['active'][f'{policy}/{group}'],flush=True)
    out['cpu_seconds']=time.process_time()-start
    return out
if __name__=='__main__':Path(__file__).with_name('STATE06_QUERY_RECEIPTS.json').write_text(json.dumps(run(),indent=2)+'\n')
