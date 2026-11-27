"""Root-local exact cut boundary index and incremental residual distances."""
class Index:
    def __init__(self,g,bounder):
        self.g=g;self.bounder=bounder
        self.head=[0]*g.n;self.prem=[0]*g.n;self.desc=[1<<h for h in range(g.n)]
        for i,b,ps,_ in bounder.rules:
            self.head[b.bit_length()-1]|=1<<i
            for h in bits(ps):self.prem[h]|=1<<i
        for h in reversed(range(g.n)):
            for i in bits(self.prem[h]):
                self.desc[h]|=self.desc[bounder.rules[i][1].bit_length()-1]
        self.zones={};self.origins={};self.order=[]
        self.unions=self.hits=self.misses=self.tests=0
    def boundary(self,z,s):
        if z in self.zones:
            self.hits+=1;return self.zones[z]
        self.misses+=1;heads=prem=0
        for h in bits(z):
            heads|=self.head[h];prem|=self.prem[h];self.unions+=2
        ids=tuple(bits(heads&~prem));assert ids
        self.zones[z]=ids;self.origins[z]=s;self.order.append(z)
        return ids
    def update(self,s,a,d,ids):
        dirty=0
        for i in ids:dirty|=self.desc[self.bounder.rules[i][1].bit_length()-1]
        d=list(d)
        for h in bits(dirty&~s):
            group=self.bounder.groups[h];self.bounder.scans+=len(group)
            d[h]=min(a[i]+max((d[p] for p in ps),default=0) for i,ps in group)
        return d
    def candidate(self,z,s,a,transfer):
        threshold=z;prior=self.order[-8:];ids=self.boundary(z,s);self.bounder.scans+=len(ids)
        delta=min(a[i] for i in ids);assert delta>0
        if transfer:
            for other in reversed(prior):
                if other==threshold or self.origins[other]==s or other&s:continue
                self.tests+=1;alt=self.zones[other];self.bounder.scans+=len(alt)
                charge=min(a[i] for i in alt)
                if charge>delta:delta,ids,z=charge,alt,other
        return z,delta,ids

def bits(mask):
    while mask:
        b=mask&-mask;yield b.bit_length()-1;mask^=b
