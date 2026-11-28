"""Canonical last-use polytope and optional MILP primal proposal."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix
from datagoblin.training_gym.spectral_face import rational_vector
from datagoblin.training_gym.spectral_quotient import score


class Model:
    def __init__(self,q):
        self.q=q; self.t,self.d=q['T'],q['d']; n=self.t*self.d; self.n=n
        rows=[]; cols=[]; vals=[]; m=0
        def add(es):
            nonlocal m
            for j,v in es:rows.append(m);cols.append(j);vals.append(v)
            m+=1
        for i in range(n):
            add([(i,1),(n+i,-1)]) # y<=h
            if i+self.d<n:
                add([(n+i+self.d,1),(n+i,-1)]) # h_next<=h
                add([(n+i,1),(n+i+self.d,-1),(i,-1)]) # h-h_next<=y
            else:add([(n+i,1),(i,-1)])
        self.a=coo_matrix((vals,(rows,cols)),shape=(m,2*n)).tocsr()
        j=np.arange(n);self.eq=coo_matrix((np.ones(n),(j//self.d,j)),shape=(self.t,2*n)).tocsr()
        self.c=np.concatenate((-np.asarray(q['u'],dtype=float).ravel(),(q['lambda']*np.diff(q['x'])[:,None]*np.asarray(q['bits'])).ravel()))

    def bounds(self,box):
        lo=np.zeros(2*self.n);hi=np.ones(2*self.n)
        for j,(a,b) in enumerate(box):
            for i in range(self.t):
                p=self.n+i*self.d+j
                if i<a:lo[p]=1
                if i>=b:hi[p]=0
        return lo,hi

    def solve(self,box):
        lo,hi=self.bounds(box)
        z=linprog(self.c,A_ub=self.a,b_ub=np.zeros(self.a.shape[0]),A_eq=self.eq,b_eq=self.q['k'],bounds=list(zip(lo,hi)),method='highs',options={'time_limit':0.25})
        return ((rational_vector(-z.eqlin.marginals),z.x[self.n:].reshape(self.t,self.d)) if z.success else (None,None))

    def primal(self,box):
        lo,hi=self.bounds(box)
        z=milp(self.c,integrality=np.r_[np.zeros(self.n),np.ones(self.n)],bounds=Bounds(lo,hi),constraints=[LinearConstraint(self.a,-np.inf,0),LinearConstraint(self.eq,self.q['k'],self.q['k'])],options={'time_limit':0.25,'node_limit':128})
        if z.x is None:return None,int(z.status)
        h=z.x[self.n:].reshape(self.t,self.d)
        if np.max(np.abs(h-np.rint(h)))>1e-5:return None,int(z.status)
        last=[int(round(sum(h[:,j]))) for j in range(self.d)]
        if any(not a<=v<=b for v,(a,b) in zip(last,box,strict=True)):return None,int(z.status)
        schedule=[]
        for i,k in enumerate(self.q['k'],1):
            req=[j for j,l in enumerate(last) if l==i]
            opt=sorted((j for j,l in enumerate(last) if l>i),key=lambda j:(-self.q['u'][i-1][j],j))
            if not 0<=k-len(req)<=len(opt):return None,int(z.status)
            schedule.append(sum(1<<j for j in req+opt[:k-len(req)]))
        score(self.q,schedule)
        return schedule,int(z.status)
