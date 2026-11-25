"""Train-only affine rule recovery and independent orbit check for STATE-05."""
import itertools,json,math
import numpy as np
from state05 import Q,NS,U,TABLE,HELD,OPS,panel

def features(s):
 return np.array([s//Q,s%Q,1],dtype=int)

def det3(m):
 a,b,c=m
 return int(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))

def fit(table):
 coeff=np.array(list(itertools.product(range(Q),repeat=3)),dtype=int);records=[];pred=np.zeros_like(table)
 X=np.array([features(s) for s in range(NS)])
 for u in range(U):
  train=np.flatnonzero(~HELD[:,u]);trip=next(t for t in itertools.combinations(train,3) if math.gcd(det3(X[list(t)]),Q)==1)
  Y=np.column_stack((table[list(trip),u]//Q,table[list(trip),u]%Q));p=(X[list(trip)]@coeff.T)%Q;cs=[]
  for j in range(2):
   matches=coeff[(p==Y[:,j,None]).all(0)];assert len(matches)==1;cs.append(matches[0])
  c=np.stack(cs);out=(X@c.T)%Q;pred[:,u]=out[:,0]*Q+out[:,1]
  records.append(dict(op=OPS[u],witness=list(map(int,trip)),det=det3(X[list(trip)]),coeff=c.tolist(),train_errors=int((pred[train,u]!=table[train,u]).sum()),held_errors=int((pred[HELD[:,u],u]!=table[HELD[:,u],u]).sum())))
 return pred,records

def run():
 pred,rec=fit(TABLE);assert (pred==TABLE).all();words=0;rank_counts={}
 for key,(starts,ops,y,_) in panel().items():
  ranks=[]
  for w,ss,trace in zip(ops[::Q],starts.reshape(-1,Q),y.reshape(-1,Q,y.shape[1])):
   A=np.eye(2,dtype=int);b=np.zeros(2,dtype=int)
   for u in w:
    c=np.array(rec[u]['coeff']);A=(c[:,:2]@A)%Q;b=(c[:,:2]@b+c[:,2])%Q
   rank=Q//math.gcd(Q,int(A[0,0]),int(A[1,0]));p=(np.array([features(s)[:2] for s in ss])@A.T+b)%Q;p=p[:,0]*Q+p[:,1];assert np.array_equal(p,trace[:,-1]);assert len(set(p))==rank;ranks.append(rank);words+=1
  rank_counts[key]={str(k):ranks.count(k) for k in (1,2,4,8)}
 # Negative control: replace one affine rule by multiplication. Use train-only validation to reject the misspecified hypothesis.
 nonlin=TABLE.copy();nonlin[:,8]=np.array([(s//Q*(s%Q)%Q)*Q+s%Q for s in range(NS)])
 _,bad=fit(nonlin);assert bad[8]['train_errors']>0
 return dict(witness_labels=3*U,additional_train_validation_labels=45*U,held_correct=int((pred[HELD]==TABLE[HELD]).sum()),held_total=int(HELD.sum()),train_correct=int((pred[~HELD]==TABLE[~HELD]).sum()),words_independently_checked=words,rank_counts=rank_counts,nonlinear_control=bad[8],rules=rec)
if __name__=='__main__':print(json.dumps(run(),indent=2))
