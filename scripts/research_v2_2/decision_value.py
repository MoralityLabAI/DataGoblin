"""Synthetic Distill-compatible adaptive-evidence decision-value pilot (CPU only)."""
from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np

SEEDS = (21, 22, 23)
SPLITS = {"train": (81, 128), "calibration": (82, 32), "iid": (83, 96), "parity": (84, 96)}


def make_world(rng, parity=False):
    n, m = 12, 6
    if parity:
        latent = np.array([[i >> j & 1 for j in range(3)] for i in range(8)] * 2)[:n]
        classes = latent[:, 0] ^ latent[:, 1] ^ latent[:, 2]
        bits = [latent[:, j] for j in range(3)]
    else:
        classes = rng.permutation([0] * 6 + [1] * 6)
        bits = []
    while len(bits) < m:
        b = rng.integers(0, 2, n)
        if len(set(b)) == 2 and not any(np.array_equal(b, x) or np.array_equal(1-b, x) for x in bits):
            bits.append(b)
    order = rng.permutation(n)
    weights = rng.integers(1, 6, n)
    return {"hypotheses": [{"weight": int(weights[i]), "class": "AB"[classes[i]]} for i in order],
            "tests": [{"id": f"t{j}", "cost": int(rng.integers(1, 4)),
                       "bits": [int(bits[k][i]) for i in order]} for j, k in enumerate(rng.permutation(m))],
            "budget": 7, "correct_reward": 12, "wrong_reward": -16}


class World:
    def __init__(self, d):
        self.d = d
        self.h = d["hypotheses"]
        self.tests = d["tests"]
        self.full = (1 << len(self.h)) - 1
        self.masks = [sum(1 << i for i, b in enumerate(t["bits"]) if b) for t in self.tests]
        self.costs = [t["cost"] for t in self.tests]
        self.work = 0

    @cache  # noqa: B019 -- cleared after each world by the campaign
    def mass(self, mask):
        return sum(h["weight"] for i, h in enumerate(self.h) if mask >> i & 1)

    @cache  # noqa: B019 -- cleared after each world by the campaign
    def a_mass(self, mask):
        return sum(h["weight"] for i, h in enumerate(self.h) if mask >> i & 1 and h["class"] == "A")

    def stop(self, mask):
        w, a = self.mass(mask), self.a_mass(mask)
        c, r = self.d["correct_reward"], self.d["wrong_reward"]
        return max((a*c+(w-a)*r, "A"), ((w-a)*c+a*r, "B"), (0, "defer"))

    def legal(self, mask, budget, used):
        for i, cost in enumerate(self.costs):
            one = mask & self.masks[i]
            if not (used >> i & 1) and cost <= budget and one and one != mask:
                yield i, mask ^ one, one

    @cache  # noqa: B019 -- cleared after each world by the campaign
    def value(self, mask, budget, used, depth):
        self.work += 1
        best = self.stop(mask)[0]
        if depth:
            for i, zero, one in self.legal(mask, budget, used):
                best = max(best, self.q(mask, budget, used, i, zero, one, depth))
        return best

    def q(self, mask, budget, used, i, zero, one, depth):
        c = self.costs[i]
        return (self.value(zero, budget-c, used | 1 << i, depth-1)
                + self.value(one, budget-c, used | 1 << i, depth-1) - c*self.mass(mask))

    def features(self, mask, budget, used, i, zero, one):
        w, z, o = self.mass(mask), self.mass(zero), self.mass(one)
        p, pz, po = self.a_mass(mask)/w, self.a_mass(zero)/z, self.a_mass(one)/o
        stop = self.stop(mask)[0]/w
        q1 = (self.stop(zero)[0]+self.stop(one)[0])/w-self.costs[i]
        c = self.costs[i]
        return [1, budget/7, c/3, (budget-c)/7, w/self.mass(self.full),
                mask.bit_count()/len(self.h), (len(self.tests)-used.bit_count())/len(self.tests),
                abs(p-.5)*2, min(z,o)/w*2, abs(pz-po),
                (abs(pz-.5)*z+abs(po-.5)*o)/w*2, stop/12, q1/12, (q1-stop)/12]

    def examples(self):
        states = {(self.full, self.d["budget"], 0)}
        for i, z, o in self.legal(self.full, self.d["budget"], 0):
            states.update(((z, self.d["budget"]-self.costs[i], 1 << i),
                           (o, self.d["budget"]-self.costs[i], 1 << i)))
        x, y = [], []
        for mask, b, used in sorted(states):
            for i, z, o in self.legal(mask, b, used):
                x.append(self.features(mask, b, used, i, z, o))
                y.append((self.q(mask, b, used, i, z, o, len(self.tests))-self.stop(mask)[0])/self.mass(mask)/12)
        return x, y

    def policy(self, mode, model=None, threshold=0.):
        decisions = 0
        def build(mask, b, used):
            nonlocal decisions
            decisions += 1
            v, stop = self.stop(mask)
            best, choice = threshold, None
            for i, z, o in self.legal(mask, b, used):
                if mode == "stop":
                    continue
                if mode in ("one_step", "two_step", "oracle"):
                    depth = {"one_step": 1, "two_step": 2, "oracle": len(self.tests)}[mode]
                    score = (self.q(mask,b,used,i,z,o,depth)-v)/self.mass(mask)/12
                elif mode == "balance_per_cost":
                    score = min(self.mass(z), self.mass(o))/self.mass(mask)/self.costs[i]
                else:
                    score = float(model.predict(np.array([self.features(mask,b,used,i,z,o)]))[0])
                if score > best:
                    best, choice = score, (i,z,o)
            if choice is None:
                return ("stop", stop)
            i,z,o = choice
            return ("test", i, build(z,b-self.costs[i],used | 1 << i),
                    build(o,b-self.costs[i],used | 1 << i))
        tree = build(self.full,self.d["budget"],0)
        return tree, decisions

    def replay(self, tree):
        total, spent_total = 0, 0
        for i,h in enumerate(self.h):
            node, spent, used = tree, 0, set()
            while node[0] == "test":
                j = node[1]
                if j in used:
                    raise ValueError("repeated test")
                used.add(j)
                spent += self.costs[j]
                if spent > self.d["budget"]:
                    raise ValueError("budget exceeded")
                node = node[2+self.tests[j]["bits"][i]]
            if node[1] == "defer":
                reward = 0
            else:
                reward = self.d["correct_reward"] if node[1] == h["class"] else self.d["wrong_reward"]
            total += h["weight"]*(reward-spent)
            spent_total += h["weight"]*spent
        return total/self.mass(self.full), spent_total/self.mass(self.full)


class Head:
    def __init__(self, seed, depth, hidden):
        rng = np.random.default_rng(seed)
        self.depth = depth
        self.params = [rng.normal(0,.2,(14,hidden)), np.zeros(hidden),
                       rng.normal(0,.08,(hidden,hidden)), rng.normal(0,.2,hidden), np.zeros(1)]
        self.calls = 0

    def forward(self,x):
        w,b,r,o,c = self.params
        h = np.zeros((len(x),len(b)))
        states = [h]
        proj = x@w+b
        for step in range(self.depth):
            h = np.tanh(proj if step == 0 else proj+h@r)
            states.append(h)
        return h@o+c, states

    def fit(self,x,y):
        rng = np.random.default_rng(991)
        p = self.params
        m = [np.zeros_like(a) for a in p]
        v = [np.zeros_like(a) for a in p]
        step = 0
        for _ in range(80):
            for batch in np.array_split(rng.permutation(len(x)), max(1,len(x)//128)):
                xb,yb=x[batch],y[batch]
                pred,h=self.forward(xb)
                e=2*(pred-yb)/len(batch)
                g=[np.zeros_like(a) for a in p]
                g[3]=h[-1].T@e
                g[4][0]=e.sum()
                dh=e[:,None]*p[3]
                for j in range(self.depth,0,-1):
                    a=dh*(1-h[j]**2)
                    g[0]+=xb.T@a
                    g[1]+=a.sum(axis=0)
                    g[2]+=h[j-1].T@a
                    dh=a@p[2].T
                step+=1
                for i in range(len(p)):
                    m[i]=.9*m[i]+.1*g[i]
                    v[i]=.999*v[i]+.001*g[i]**2
                    p[i]-=.003*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
        return self

    def predict(self,x):
        self.calls += len(x)
        return self.forward(x)[0]


def score(worlds, mode, model=None, threshold=0.):
    rows=[]
    for d in worlds:
        # Fresh solver for every arm. Oracle is evaluator-only.
        w=World(d)
        calls=model.calls if model is not None else 0
        tree,n=w.policy(mode,model,threshold)
        u,c=w.replay(tree)
        work=w.work
        optimum=w.value(w.full,d["budget"],0,len(w.tests))/w.mass(w.full)
        if u>optimum+1e-9 or (mode=="oracle" and abs(u-optimum)>1e-9):
            raise ValueError("replay differs from exact DP")
        rows.append([u,optimum-u,c,n,work,model.calls-calls if model is not None else 0])
        World.mass.cache_clear()
        World.a_mass.cache_clear()
        World.value.cache_clear()
    return np.array(rows)


def run(worlds=None):
    if worlds is None:
        worlds={name:[make_world(rng,name=="parity") for _ in range(n)]
                for name,(seed,n) in SPLITS.items() for rng in [np.random.default_rng(seed)]}
    inventories=[{json.dumps(d,sort_keys=True) for d in rows} for rows in worlds.values()]
    if any(inventories[i]&inventories[j] for i in range(len(inventories)) for j in range(i)):
        raise ValueError("split overlap")
    x,y=[],[]
    for d in worlds["train"]:
        a,b=World(d).examples()
        x.extend(a)
        y.extend(b)
        World.mass.cache_clear()
        World.a_mass.cache_clear()
        World.value.cache_clear()
    x,y=np.array(x),np.array(y)
    result={"schema":"research-v2.2-decision-value-local-v1", "numpy":np.__version__,
            "train_action_examples":len(x),"splits":{k:len(v) for k,v in worlds.items()},"arms":{}, "paired":{}}
    outcomes = {}
    names=("utility","regret","credits","tree_nodes","dp_cache_misses","learned_scores")
    for split in ("iid","parity"):
        if split not in worlds:
            continue
        result["arms"][split]={}
        outcomes[split] = {}
        for mode in ("stop","balance_per_cost","one_step","two_step","oracle"):
            rows = score(worlds[split],mode)
            outcomes[split][mode] = rows[:,0]
            result["arms"][split][mode]=dict(zip(names,rows.mean(axis=0).tolist(),strict=True))
    for mode,depth,width in (("ff",1,16),("trm",3,16),("wide_ff",1,48)):
        for seed in SEEDS:
            model=Head(seed,depth,width).fit(x,y)
            tau=max((float(score(worlds["calibration"],mode,model,t)[:,0].mean()),-t,t)
                    for t in (0.,.025,.05,.1,.2))[2]
            for split in result["arms"]:
                r=score(worlds[split],mode,model,tau)
                outcomes[split].setdefault(mode,[]).append(r[:,0])
                record = {"seed":seed,"threshold":tau,**dict(zip(names,r.mean(axis=0).tolist(),strict=True))}
                result["arms"][split].setdefault(mode,[]).append(record)
    for split, arms in outcomes.items():
        means = {name: np.asarray(values).reshape(-1,len(worlds[split])).mean(axis=0)
                 for name,values in arms.items()}
        rng = np.random.default_rng(20261128)
        indices = rng.integers(len(worlds[split]), size=(10000,len(worlds[split])))
        result["paired"][split] = {}
        for control in ("one_step","two_step","ff","wide_ff"):
            d = means["trm"]-means[control]
            result["paired"][split]["trm_vs_"+control] = {"delta":float(d.mean()),
                "ci95":np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist()}
    result["scope"] = "Synthetic Distill-compatible worlds; fixed-seed root bootstrap; exploratory only."
    result["effective_head_macs"] = {"ff":240,"trm":752,"wide_ff":720}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    report=json.dumps(run(),indent=2,allow_nan=False)
    if args.out:
        if args.out.exists():
            raise ValueError("fresh output required")
        args.out.write_text(report+"\n")
    print(report)


if __name__=="__main__":
    main()
