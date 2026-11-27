"""MM-GYM-03: scoped prefix checks, budgeted ranking, and compact repair."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

SOURCE = "9c4bcad276eab5f8b17b231a4543cfe185c19c95cfb7940780dd1de265610c12"
INPUTS = "f0efbbd745eb880b53367086dd2f277e8f72621b2cf4a7d2bfebecc947da03ab"
ORACLE = "3e1b4410e3fab818811549f353ae50314985e80125f101f2f52804a19b23b59b"
SPLITS = ("train", "dev", "public_challenge")
SEEDS = (109, 113, 127)
DEPTHS = (1, 4, 8)

def require(cond, msg):
    if not cond:
        raise ValueError(msg)

def load(root):
    receipt = json.loads((root/"receipt.json").read_text())
    require(receipt["schema"] == "mm-gym-02-v1" and receipt["source_sha256"] == SOURCE,
            "source receipt mismatch")
    for name, expected in (("inputs.jsonl", INPUTS), ("oracle.jsonl", ORACLE)):
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
                == receipt["artifact_sha256"][name], "frozen artifact mismatch")
    x = [json.loads(s) for s in (root/"inputs.jsonl").read_text().splitlines()]
    ys = [json.loads(s) for s in (root/"oracle.jsonl").read_text().splitlines()]
    y = {a["id"]:a for a in ys}
    require(len(x) == len(y) == len(ys) == 585 and {a["id"] for a in x} == set(y),
            "root coverage")
    require(Counter(a["split"] for a in x) ==
            {"train":186,"dev":199,"public_challenge":200}, "split coverage")
    return x, y

def apply(x, label):
    """One scoped assertion application; no suffix validation or new labels."""
    s = x["candidate_signatures"][label]
    f, e, stack = s["floating"], s["essential"], x["prefix_stack"]
    n = len(f)+len(e)
    if len(stack) < n:
        return None
    args, sub = stack[-n:] if n else [], {}
    for (typ,var), v in zip(f,args[:len(f)], strict=True):
        if not v or v[0] != typ:
            return None
        sub[var] = v[1:]
    def inst(stmt):
        return [v for t in stmt for v in sub.get(t,[t])]
    for h,v in zip(e,args[len(f):], strict=True):
        if inst(h) != v:
            return None
    variables = set(x["variables"])
    dv = {tuple(sorted(z)) for z in x["distinct"]}
    for a,b in s["distinct"]:
        for v in variables.intersection(sub[a]):
            for w in variables.intersection(sub[b]):
                if v == w or tuple(sorted((v,w))) not in dv:
                    return None
    return inst(s["statement"])

def similarity(a,b):
    ca,cb = Counter(a),Counter(b)
    return 2*sum((ca&cb).values())/max(1,len(a)+len(b))

def banks(rows, oracle):
    frequency = Counter(oracle[x["id"]]["reference_action"] for x in rows if x["split"]=="train")
    out = {}
    for x in rows:
        id_, bank = x["id"], x["candidates"]
        z = {a:apply(x,a) for a in bank}
        legal = [a for a in bank if z[a] is not None]
        require(set(legal) == set(oracle[id_]["immediate_legal_actions"]), "legality disagreement")
        require(set(oracle[id_]["accepted_actions"]) <= set(legal) and legal, "oracle contradiction")
        rank = sorted(legal, key=lambda a:(-(z[a]==x["target"]),
                      -similarity(z[a],x["target"]),-frequency[a],a))
        out[id_] = (z, {"bank":bank,"frequency":sorted(bank,key=lambda a:(-frequency[a],a)),
                        "prefix_legal":legal,"symbolic":rank})
    return out, frequency

def metrics(rows, oracle, rankings):
    results = {}
    for split in SPLITS:
        subset = [x for x in rows if x["split"] == split]
        n = len(subset)
        hits, calls = Counter(), Counter()
        for x in subset:
            valid = set(oracle[x["id"]]["accepted_actions"])
            ranked = rankings[x["id"]]
            for k in (1,3,5):
                prefix = ranked[:k]
                hits[str(k)] += bool(valid.intersection(prefix))
                calls[str(k)] += next((i+1 for i,a in enumerate(prefix) if a in valid),len(prefix))
        results[split] = {"n":n,"hits":dict(hits),"hypothetical_full_replays":dict(calls)}
    return results

def feature(x, label, value, frequency):
    """Only public state and one charged local application are used."""
    s, target, stack = x["candidate_signatures"][label], x["target"], x["prefix_stack"]
    statement = s["statement"]
    last = stack[-1] if stack else []
    v = value or []
    def log_length(z):
        return math.log1p(len(z))/6
    def common(a,b):
        return sum(1 for p,q in zip(a,b, strict=False) if p==q)/max(1,len(a),len(b))
    return [
        float(value is not None), float(v == target), similarity(v,target), common(v,target),
        similarity(statement,target), common(statement,target), similarity(v,last),
        similarity(statement,last), log_length(v), log_length(target), log_length(statement), log_length(stack), log_length(x["prefix"]),
        log_length(x["suffix"]), log_length(s["floating"]), log_length(s["essential"]), log_length(s["distinct"]),
        math.log1p(frequency[label])/6, float(s["kind"]=="$a"),
        float(not x["suffix"]), float(label in x["prefix"]), float(label in x["suffix"]),
    ]

def neural(rows, oracle, cache, frequency):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fs, masks, ys = [], [], []
    for x in rows:
        z,_ = cache[x["id"]]
        bank = x["candidates"]
        valid = set(oracle[x["id"]]["accepted_actions"])
        vecs = [feature(x,a,z[a],frequency) for a in bank]
        fs.append(vecs+[[0.]*len(vecs[0])]*(12-len(bank)))
        masks.append([z[a] is not None for a in bank]+[False]*(12-len(bank)))
        ys.append([a in valid for a in bank]+[False]*(12-len(bank)))
    X, M, Y = torch.tensor(fs), torch.tensor(masks), torch.tensor(ys)
    train = torch.tensor([i for i,x in enumerate(rows) if x["split"]=="train"])
    class Ranker(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(X.shape[-1],32)
            self.cell = nn.Linear(64,32)
            self.out = nn.Linear(32,1)
        def forward(self, x, m, depth):
            h = torch.tanh(self.enc(x))
            for _ in range(depth):
                pool = (h*m.unsqueeze(-1)).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1).unsqueeze(-1)
                h = h + 0.5*torch.tanh(self.cell(torch.cat((h,pool.expand_as(h)),dim=-1)))
            return self.out(h).squeeze(-1).masked_fill(~m,-1e9)
    results = []
    for seed in SEEDS:
        for depth in DEPTHS:
            torch.manual_seed(seed)
            model = Ranker()
            opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
            gen = torch.Generator().manual_seed(seed+10000)
            start = time.perf_counter()
            updates = 0
            for _ in range(80):
                for batch in train[torch.randperm(len(train), generator=gen)].split(16):
                    scores = model(X[batch],M[batch],depth)
                    loss = (torch.logsumexp(scores,dim=1)-
                            torch.logsumexp(scores.masked_fill(~Y[batch],-1e9),dim=1)).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    updates += 1
            with torch.no_grad():
                scores = model(X,M,depth)
            ranks = {}
            for i,x in enumerate(rows):
                bank = x["candidates"]
                ranks[x["id"]] = sorted((a for j,a in enumerate(bank) if M[i,j]),
                                        key=lambda a:(-float(scores[i,bank.index(a)]),a))
            results.append({"seed":seed,"depth":depth,"parameters":sum(p.numel() for p in model.parameters()),
                            "updates":updates,"wall_seconds":round(time.perf_counter()-start,4),
                            "metrics":metrics(rows,oracle,ranks)})
    return results

def run(root, with_neural):
    rows, oracle = load(root)
    cache, freq = banks(rows,oracle)
    baseline = {name:metrics(rows,oracle,{x["id"]:cache[x["id"]][1][name] for x in rows})
                for name in ("bank","frequency","prefix_legal","symbolic")}
    local = {s:sum(len(x["candidates"]) for x in rows if x["split"]==s) for s in SPLITS}
    uniform = {s:sum(len(oracle[x["id"]]["accepted_actions"])/len(x["candidates"])
                     for x in rows if x["split"]==s)/sum(x["split"]==s for x in rows) for s in SPLITS}
    receipt = {"schema":"mm-gym-03-v1","source_sha256":SOURCE,"input_sha256":INPUTS,
               "oracle_sha256":ORACLE,"local_checks":local,"legality_disagreements":0,
               "uniform_top1_expected":uniform,"baseline":baseline,
               "neural":neural(rows,oracle,cache,freq) if with_neural else [],
               "scope":"public source-order closed-bank pilot; oracle-scored rankings, no live suffix calls"}
    return receipt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle",type=Path)
    p.add_argument("--neural",action="store_true")
    p.add_argument("--out",type=Path)
    a = p.parse_args()
    receipt = json.dumps(run(a.bundle,a.neural),indent=2,sort_keys=True)+"\n"
    if a.out:
        a.out.write_text(receipt)
    print(receipt)

if __name__=="__main__":
    main()
