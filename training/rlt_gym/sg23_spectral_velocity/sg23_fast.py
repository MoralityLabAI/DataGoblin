"""Incremental active-box score maintenance during a coordinate sweep."""
from sg23_cone import Cone
from datagoblin.training_gym.spectral_quotient import indices


class FastCone(Cone):
    def sweep(self, v):
        v=list(v); scores=[]
        for j,boxes in enumerate(self.boxes):
            z=[a-b for a,b in zip(self.du[j],v,strict=True)]
            scores.append([sum(z[i] for i in indices(a))+sum(max(0,z[i]) for i in indices(b^a))
                           -self.dp[j][a.bit_length()] for a,b in boxes])
        for i in range(self.t-1,-1,-1):
            bit=1<<i; thresholds=[];forced=0
            for j,boxes in enumerate(self.boxes):
                z=self.du[j][i]-v[i];absent=present=None
                for (a,b),score in zip(boxes,scores[j],strict=True):
                    val=score-(z if a&bit else max(0,z) if b&bit else 0)
                    if not a&bit:absent=val if absent is None else max(absent,val)
                    if b&bit:
                        w=val+self.du[j][i]
                        present=w if present is None else max(present,w)
                if absent is None:forced+=1
                elif present is not None:thresholds.append(present-absent)
            k=self.anchor.q.k[i]-forced
            assert 0<=k<=len(thresholds)
            thresholds.sort(reverse=True);old=v[i]
            if k<len(thresholds):v[i]=max(v[i],thresholds[k])
            if k:v[i]=min(v[i],thresholds[k-1])
            if v[i]!=old:
                for j,boxes in enumerate(self.boxes):
                    before=self.du[j][i]-old;after=self.du[j][i]-v[i]
                    for h,(a,b) in enumerate(boxes):
                        if a&bit:scores[j][h]+=after-before
                        elif b&bit:scores[j][h]+=max(0,after)-max(0,before)
        return v
