"""Numerical LP proposal generator. LP status is never a proof."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from datagoblin.training_gym.spectral_face import rational_vector


def propose(q):
    T,d=q['T'],q['d']
    # eta_i free; p_ij>=max(0,u_ij-eta_i); w_j>=all last-use expressions.
    n=T+T*d+d
    ii,jj,v,rhs=[],[],[],[]
    def row(entries,b):
        r=len(rhs);rhs.append(b)
        for j,a in entries:
            ii.append(r);jj.append(j);v.append(a)
    for i in range(T):
        for j in range(d):
            row([(i,-1),(T+i*d+j,-1)],-q['u'][i][j])
            row([(i,-1),(T+T*d+j,-1),*[(T+l*d+j,1) for l in range(i)]],
                q['lambda']*q['bits'][j]*(q['x'][i+1]-q['x'][0])-q['u'][i][j])
    A=coo_matrix((v,(ii,jj)),shape=(len(rhs),n)).tocsr()
    c=np.asarray(q['k']+[0]*(T*d)+[1]*d,dtype=float)
    z=linprog(c,A_ub=A,b_ub=rhs,bounds=[(None,None)]*T+[(0,None)]*(T*d+d),method='highs',
              options={'time_limit':2.0})
    if not z.success:
        return None,dict(status=int(z.status),message=str(z.message))
    eta,den=rational_vector(z.x[:T])
    return (eta,den),dict(status=int(z.status),objective=float(z.fun),iterations=int(z.nit))
