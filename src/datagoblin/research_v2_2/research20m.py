"""DG-Research20M: token-budgeted, verified research episodes; CPU generation only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from functools import cache
from itertools import product
from pathlib import Path

from . import research_pretrain as v1

V1_POSITIONS = 4_221_281
TARGET = 20_000_000 - V1_POSITIONS
QUOTAS = dict(adaptive_evidence=3_000_000, robust_discrimination=3_000_000,
              research_portfolio=2_800_000, proof_obligations=2_500_000,
              paired_causal_gate=2_500_000, research_memory=1_978_719)
PROFILES = (
    ("adaptive_evidence", "parity_synergy"), ("adaptive_evidence", "dependent_tests"),
    ("robust_discrimination", "shared_controls"), ("robust_discrimination", "bundled_controls"),
    ("research_portfolio", "scenario_conflict"), ("research_portfolio", "resource_bottleneck"),
    ("proof_obligations", "unlock_chain"), ("proof_obligations", "refutation_barrier"),
    ("paired_causal_gate", "unequal_repeats"), ("paired_causal_gate", "confounded_budget"),
    ("research_memory", "offline_cache"), ("research_memory", "probe_alias"),
)
SOURCE = "01b5ce2522175461b3ddc1a4113f0c5683339610"
SEED = 2026871000
PILOT_SEED = 2026891000


def prior_roots(gen, manifest):
    """Replay v1 membership and compare exact published split inventory hashes."""
    if manifest["manifest_id"] != v1.digest({k: v for k, v in manifest.items() if k != "manifest_id"}):
        raise ValueError("v1 manifest binding")
    seen = v1.old_roots(gen)
    if len(seen) != manifest["known_excluded_roots"]:
        raise ValueError("old inventory mismatch")
    for split in ("train", "development"):
        roots = []
        if manifest["splits"][split]["rejected"].get("over_context_views", 0):
            raise ValueError("v1 requires archived member inventory")
        for fi, kind in enumerate(v1.FAMILIES):
            count = manifest["splits"][split]["families"][kind]["roots"]
            accepted, i = 0, 0
            seed = v1.SEEDS[split]+fi*1000
            while accepted < count:
                if i >= max(128,count*16):
                    raise ValueError("v1 replay exhausted")
                d = (v1.memory_world(seed,i) if kind=="research_memory"
                     else gen(kind,seed,i,v1.DIFFICULTIES[i%3]))
                i += 1
                root = v1.semantic(kind,d)
                if root in seen:
                    continue
                seen.add(root)
                roots.append(root)
                accepted += 1
        raw = (json.dumps(sorted(roots),indent=2)+"\n").encode()
        if hashlib.sha256(raw).hexdigest() != manifest["splits"][split]["files"]["roots.json"]["sha256"]:
            raise ValueError("v1 root inventory hash mismatch")
    return seen


def world(gen, kind, profile, seed, i):
    rng = random.Random(seed*1000003+i)
    if profile == "parity_synergy":
        m = rng.choice((3,4))
        xs = list(product((0,1),repeat=m))
        selected = rng.sample(range(m),rng.randint(2,m))
        hs = [dict(weight=rng.randint(1,5),**{"class": "B" if sum(x[j] for j in selected)%2 else "A"})
              for x in xs]
        tests = [dict(id=f"t{j}",cost=rng.randint(1,3),bits=[x[j] for x in xs]) for j in range(m)]
        return dict(hypotheses=hs,tests=tests,budget=rng.randint(3,10),
                    correct_reward=rng.randint(8,20),wrong_reward=-rng.randint(12,28))
    if profile == "offline_cache":
        n = rng.randint(3,5)
        requests = [rng.randrange(n) for _ in range(rng.randint(12,24))]
        return dict(kind=profile,requests=requests,sizes=[rng.randint(1,4) for _ in range(n)],
                    rents=[rng.randint(2,12) for _ in range(n)],buys=[rng.randint(2,18) for _ in range(n)],
                    capacity=rng.randint(2,8))
    if profile == "probe_alias":
        period = rng.randint(4,10)
        offset = rng.randrange(period-1)
        length = rng.randint(32,72)
        repeats = rng.randint(1,4)
        keys = [100+t if (t%period in (offset,offset+1) or rng.random()<.1) else t//period%repeats
                for t in range(length)]
        return dict(kind=profile,keys=keys,period=period,offset=offset,window=rng.randint(8,24))
    d = gen(kind,seed,i,("standard","frontier")[i%2])
    if profile == "dependent_tests":
        x,y = rng.sample(d["tests"],2)
        d["tests"].append(dict(id="derived",cost=rng.randint(1,4),
                               bits=[a^b for a,b in zip(x["bits"],y["bits"],strict=True)]))
        d["budget"] = min(12,d["budget"]+rng.randrange(3))
    elif kind == "robust_discrimination":
        controls = d["experiments"][:2]
        for c in controls:
            c["cost"] = rng.randint(2,7)
        experiments = d["experiments"][2:]
        for j,e in enumerate(experiments):
            e["requires"] = sorted(set(e["requires"]+[controls[j%2]["id"]]))
        if profile == "bundled_controls":
            for a,b in zip(experiments[::2],experiments[1::2],strict=True):
                a["requires"] = sorted(set(a["requires"]+[b["id"]]))
                b["requires"] = sorted(set(b["requires"]+[a["id"]]))
        d["budget"] += rng.randint(0,6)
    elif profile == "scenario_conflict":
        for j,r in enumerate(d["jobs"][1:]):
            r["gains"][j%3] = rng.randint(6,14)
            r["gains"][(j+1)%3] = -rng.randint(1,8)
    elif profile == "resource_bottleneck":
        for j,r in enumerate(d["jobs"][1:],1):
            if j>1:
                r["requires"] = sorted(set(r["requires"]+[d["jobs"][j-1]["id"]]))
        d["ram_cap"] = rng.randint(3,6)
    elif kind == "proof_obligations":
        claims = d["claims"]
        for j,c in enumerate(claims[1:],1):
            c["requires"] = sorted(set(c["requires"]+[claims[j-1]["id"]]))
            if profile == "unlock_chain":
                c["state"] = "open" if j%3==0 else "checked"
        if profile == "refutation_barrier":
            claims[2]["state"] = "refuted"
    elif profile == "unequal_repeats":
        for a in d["arms"]:
            for r in a["roots"]:
                if rng.randrange(2):
                    r["active"] = r["active"]*rng.randint(2,4)
                else:
                    r["sham"] = r["sham"]*rng.randint(2,4)
    elif profile == "confounded_budget":
        a = rng.choice(d["arms"])
        r = rng.choice(a["roots"])
        r["active_dose"] += rng.randint(1,3)
    return d


def revised(kind,d,rng):
    out = copy.deepcopy(d)
    if kind == "adaptive_evidence":
        t = rng.choice(out["tests"])
        t["cost"] += 1
        change = f"New measurement cost for {t['id']}: {t['cost']}."
    elif kind == "robust_discrimination":
        out["budget"] = max(0,out["budget"]-rng.randint(1,4))
        change = f"Revised experiment budget: {out['budget']}."
    elif kind == "research_portfolio":
        out["cpu_budget"] -= rng.randint(1,3)
        change = f"Revised CPU budget: {out['cpu_budget']}."
    elif kind == "proof_obligations":
        c = next(c for c in out["claims"] if c["state"] != "refuted")
        c["state"] = "refuted"
        change = f"New counterexample refutes claim {c['id']}."
    elif kind == "paired_causal_gate":
        r = rng.choice(rng.choice(out["arms"])["roots"])
        r["active_visits"] += 1
        change = "A hypothetical new execution receipt records an additional active visit on one root."
    elif d["kind"] == "offline_cache":
        out["capacity"] += 1
        change = f"Revised serialized capacity: {out['capacity']}."
    else:
        out["offset"] = (out["offset"]+1)%(out["period"]-1)
        change = f"Move the two probe positions to offset {out['offset']} and the next slot."
    return out,change


def cache_replay(d,masks):
    if len(masks)!=len(d["requests"]):
        raise ValueError("wrong cache trace length")
    old,total = 0,0
    n = len(d["sizes"])
    for k,new in zip(d["requests"],masks,strict=True):
        bit = 1 << k
        if type(new) is not int or not 0<=new<1 << n:
            raise ValueError("cache mask")
        if sum(s for i,s in enumerate(d["sizes"]) if new>>i&1)>d["capacity"]:
            raise ValueError("capacity")
        if old&bit:
            if new!=old:
                raise ValueError("hit trace must retain state")
        else:
            total += d["rents"][k]
            if new!=old:
                if not new&bit or new&~(old|bit):
                    raise ValueError("unsupported admission")
                total += d["buys"][k]
        old = new
    return total


def cache_exact(d):
    n = len(d["sizes"])
    feasible = [m for m in range(1<<n)
                if sum(s for i,s in enumerate(d["sizes"]) if m>>i&1)<=d["capacity"]]
    req = d["requests"]

    @cache
    def dp(t,m):
        if t==len(req):
            return 0,()
        bit = 1<<req[t]
        rent = d["rents"][req[t]]
        options = [(m,0 if m&bit else rent)]
        if not m&bit:
            options.extend((u,rent+d["buys"][req[t]]) for u in feasible if u&bit and not u&~(m|bit))
        return min((cost+dp(t+1,u)[0],(u,*dp(t+1,u)[1])) for u,cost in options)

    cost,path = dp(0,0)
    # Independent forward state enumeration. Hit/bypass keep the resident
    # set; free evictions can be postponed to a later paid admission.
    frontier = {0:0}
    for k in req:
        nxt = {}
        bit = 1<<k
        for old,value in frontier.items():
            for target in feasible:
                if old&bit:
                    if target!=old:
                        continue
                    delta = 0
                elif target==old:
                    delta = d["rents"][k]
                elif target&bit and not target&~(old|bit):
                    delta = d["rents"][k]+d["buys"][k]
                else:
                    continue
                nxt[target] = min(nxt.get(target,10**9),value+delta)
        frontier = nxt
    if cost!=min(frontier.values()) or cost!=cache_replay(d,path):
        raise ValueError("offline optimizer disagreement")
    return dict(cost=cost,resident_masks=list(path))


def cache_delayed(d):
    resident,seen,path = [],set(),[]
    for k in d["requests"]:
        if k in resident:
            resident.remove(k)
            resident.append(k)
        elif k in seen and d["sizes"][k]<=d["capacity"]:
            while sum(d["sizes"][i] for i in resident)+d["sizes"][k]>d["capacity"]:
                resident.pop(0)
            resident.append(k)
        seen.add(k)
        path.append(sum(1<<i for i in resident))
    return dict(cost=cache_replay(d,path),resident_masks=path)


def probe_exact(d):
    keys,period,offset,window = (d[k] for k in ("keys","period","offset","window"))
    last,observed = {},{}
    full,sparse,probes = 0,0,0
    for t,key in enumerate(keys):
        full += int(key in last and t-last[key]<=window)
        last[key] = t
        if t%period in (offset,offset+1):
            probes += 1
            sparse += int(key in observed and t-observed[key]<=window)
            observed[key] = t
    times = [i for i in range(len(keys)) if i%period in (offset,offset+1)]
    check_full = sum(any(keys[j]==keys[i] for j in range(max(0,i-window),i)) for i in range(len(keys)))
    check_sparse = sum(any(keys[j]==keys[i] and 0<i-j<=window for j in times) for i in times)
    if (full,sparse,probes)!=(check_full,check_sparse,len(times)):
        raise ValueError("probe replay")
    return dict(full_repeat_hints=full,sampled_repeat_hints=sparse,probes=probes,missed=full-sparse)


def memory_rows(d,after,change):
    solve = cache_exact if d["kind"]=="offline_cache" else probe_exact
    a,b = solve(d),solve(after)
    text = "Stipulated research memory model: "+v1.canonical(d)
    if d["kind"]=="offline_cache":
        instruction = ("On a miss pay rent. Admission also pays buy cost. Evict freely at admission, "
                       "then retain the requested key. Hits cost zero. No prefetch. Minimize total cost.")
        control = cache_delayed(d)
        bad = dict(a,cost=a["cost"]+1)
        if cache_replay(d,bad["resident_masks"])==bad["cost"]:
            raise ValueError("corrupt cache trace accepted")
        try:
            previous = cache_replay(after,a["resident_masks"])
        except ValueError:
            previous = None
        compare = dict(delayed=control,optimal=a,regret=control["cost"]-a["cost"])
        repair = dict(verdict="reject",reason="The reported total differs from the replayed trace.",repair=a)
        revision = dict(previous_cost_under_new_capacity=previous,new_optimal=b)
    else:
        instruction = ("Count repeat hints within the given window. Full observation sees every key; "
                       "sparse observation updates history only on the two probe slots in each block.")
        control = dict(a)
        bad = dict(a,sampled_repeat_hints=a["full_repeat_hints"]+1)
        compare = dict(full=a["full_repeat_hints"],sparse=a["sampled_repeat_hints"],missed=a["missed"],
                       scope="deterministic schedule and exact key equality")
        repair = dict(verdict="reject",reason="Sparse observations cannot contain more repeat hints than full history.",
                      repair=a)
        revision = dict(previous=a,revised=b)
    witness = dict(method="independent_finite_replay",model=d["kind"],control=control)
    return [
        ("decision",instruction+"\n"+text+"\nCompute the exact decision and ledger.",a,witness),
        ("repair",instruction+"\n"+text+"\nAudit: "+v1.canonical(bad),repair,witness),
        ("matched_control",instruction+"\n"+text+"\nCompare full and restricted methods.",compare,witness),
        ("premise_revision",instruction+"\n"+change+"\nRevised model: "+v1.canonical(after)
         +"\nPrevious answer: "+v1.canonical(a)+"\nRecompute.",revision,witness),
    ]


def views(kind,d,after,change,assess):
    if kind=="research_memory":
        return memory_rows(d,after,change)
    rows = v1.native_views(kind,d,assess)
    old = rows[0][2]
    new,witness = v1.solve_native(kind,after)
    case = dict(kind=kind,data=after)
    previous = assess(case,old)
    current = assess(case,new)
    if not (current["optimal"] and current["feasible"] and current["certificate_consistent"]):
        raise ValueError("revision target rejected")
    rows.append(("premise_revision",v1.INSTRUCTIONS[kind]+"\n"+change+"\nRevised world: "
                 +v1.canonical(v1.compact_world(kind,after))+"\nPrevious certificate: "+v1.canonical(old)
                 +"\nAudit the previous certificate under the new premises and return an optimal one.",
                 dict(previous_grade=v1.grade_summary(previous),revised_answer=new),
                 dict(**witness,checker="Distill.research_oracle.assess",
                      previous_grade=v1.grade_summary(previous),grade=v1.grade_summary(current))))
    return rows


def quota_plan(target):
    # Allocate to family, then split each family's budget evenly by profile.
    floors = {k: target*v//TARGET for k,v in QUOTAS.items()}
    floors["research_memory"] += target-sum(floors.values())
    return {profile: floors[k]//2 + (floors[k]%2 if i%2 else 0)
            for i,(k,profile) in enumerate(PROFILES)}


def build(out,gen,assess,tok,manifest,target=TARGET,pilot=False,prior=(),dev_per_profile=16,
          max_roots_per_profile=6000):
    if tok is None or not 12_000<=target<=TARGET or not 1<=max_roots_per_profile<=12000:
        raise ValueError("frozen tokenizer and bounded budget required")
    forbidden = prior_roots(gen,manifest)
    forbidden.update(prior)
    seen = set(forbidden)
    out.mkdir(parents=True,exist_ok=False)
    plan = quota_plan(target)
    result = dict(schema="dg-research20m.v1",base_manifest=manifest["manifest_id"],source_commit=SOURCE,
                  upstream=v1.UPSTREAM,tokenizer_sha256=v1.TOKENIZER,pilot=pilot,
                  target_additional_positions=target,prior_excluded=len(forbidden),
                  max_views=4,training_admission=False,training_ready=False,production_launch_authorized=False,
                  profiles=plan,splits={})
    transcript_ids = set()
    for split in ("train","development"):
        folder = out/split
        folder.mkdir()
        members, lineages = [],[]
        stats = {}
        for p,(kind,profile) in enumerate(PROFILES):
            s = Counter()
            limit = (8 if split=="train" else 2) if pilot else (
                max_roots_per_profile if split=="train" else dev_per_profile)
            quota = plan[profile] if split=="train" and not pilot else None
            seed = (PILOT_SEED if pilot else SEED)+100*p+(100000 if split=="development" else 0)
            with (folder/f"{profile}.records.jsonl").open("w",encoding="ascii") as raw, \
                 (folder/f"{profile}.packed.jsonl").open("w",encoding="ascii") as packed:
                for i in range(limit*32):
                    if s["roots"]>=limit or (quota is not None and s["positions"]>=quota):
                        break
                    d = world(gen,kind,profile,seed,i)
                    after,change = revised(kind,d,random.Random(seed+i+123))
                    a,b = v1.semantic(kind,d),v1.semantic(kind,after)
                    if a==b or a in seen or b in seen:
                        s["excluded"] += 1
                        continue
                    lineage = v1.digest([kind,a,b])
                    rows = views(kind,d,after,change,assess)
                    pending = []
                    for name,prompt,answer,witness in rows:
                        target_text = v1.canonical(answer)
                        item = v1.pack(tok,prompt,target_text)
                        if item is None:
                            break
                        text_sha = hashlib.sha256((prompt+v1.DELIMITER+target_text+"\n").encode("ascii")).hexdigest()
                        record = dict(split=split,skill="custom_research",lane="custom_research",
                                      family=kind,profile=profile,semantic_key=lineage,view=name,
                                      record_id=v1.digest([lineage,name]),world_ids=[a,b],
                                      prompt=prompt,target=target_text,
                                      delimiter=v1.DELIMITER,objective="checked_response",record_sha256=text_sha,
                                      origin=dict(seed=seed,ordinal=i,upstream=v1.UPSTREAM,source=SOURCE),
                                      witness=witness,training_admission=False,custody="candidate")
                        pending.append((record,item))
                    if len(pending)!=4:
                        s["overlength_lineages"] += 1
                        continue
                    seen.update((a,b))
                    members.extend((a,b))
                    lineages.append(lineage)
                    s["roots"] += 1
                    for record,item in pending:
                        th = record["record_sha256"]
                        if th in transcript_ids:
                            raise ValueError("duplicate transcript")
                        transcript_ids.add(th)
                        raw.write(v1.canonical(record)+"\n")
                        packed.write(v1.canonical(dict(**item,root=lineage,world_ids=[a,b],family=kind,
                            profile=profile,split=split,lane="custom_research",objective="checked_response",
                            presentation=1,record_id=record["record_id"],receipt_sha256=v1.digest(record),
                            training_admission=False))+"\n")
                        s["records"] += 1
                        s["positions"] += len(item["token_ids"])
                        s["loss_tokens"] += sum(item["target_mask"])
                        s["max_context"] = max(s["max_context"],len(item["token_ids"]))
                        s["view_"+record["view"]] += 1
            if not s["roots"] or (quota is None and s["roots"]!=limit):
                raise ValueError("profile generation shortfall")
            s["quota"] = quota or 0
            s["shortfall"] = max(0,quota-s["positions"]) if quota else 0
            stats[profile] = dict(s)
            print("DG20M_PROFILE "+v1.canonical(dict(split=split,profile=profile,**s)),flush=True)
        (folder/"members.json").write_text(json.dumps(sorted(members),indent=2)+"\n")
        (folder/"lineages.json").write_text(json.dumps(sorted(lineages),indent=2)+"\n")
        result["splits"][split] = dict(profiles=stats,roots=len(lineages),member_worlds=len(members),
            records=sum(s["records"] for s in stats.values()),positions=sum(s["positions"] for s in stats.values()),
            loss_tokens=sum(s["loss_tokens"] for s in stats.values()),
            shortfall=sum(s["shortfall"] for s in stats.values()),
            files={p.name:dict(bytes=p.stat().st_size,sha256=v1.file_hash(p)) for p in sorted(folder.iterdir())})
    if len(seen)-len(forbidden)!=sum(x["member_worlds"] for x in result["splits"].values()):
        raise ValueError("split overlap")
    result["candidate_cumulative_positions"] = V1_POSITIONS+result["splits"]["train"]["positions"]
    result["manifest_id"] = v1.digest(result)
    for folder in (out,out/"train",out/"development"):
        (folder/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print("DG20M_RECEIPT "+v1.canonical(result),flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distill-checkout",type=Path,required=True)
    p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--v1-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--target-positions",type=int,default=TARGET)
    p.add_argument("--exclude-members",type=Path,action="append",default=[])
    p.add_argument("--dev-per-profile",type=int,default=16)
    p.add_argument("--max-roots-per-profile",type=int,default=6000)
    a = p.parse_args()
    if not 1<=a.dev_per_profile<=128:
        raise ValueError("development root bound")
    path = a.distill_checkout.resolve()
    sha = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    dirty = subprocess.check_output(["git","-C",str(path),"status","--porcelain"],text=True).strip()
    if sha!=v1.UPSTREAM or dirty:
        raise ValueError("clean pinned Distill checkout required")
    if not a.pilot and not a.exclude_members:
        raise ValueError("full generation requires the spent pilot member inventories")
    prior = set()
    for f in a.exclude_members:
        rows = json.loads(f.read_text())
        if not isinstance(rows,list) or not all(isinstance(x,str) and len(x)==64 for x in rows):
            raise ValueError("malformed exclusion inventory")
        prior.update(rows)
    sys.path.insert(0,str(path))
    from Distill.research_cases import _data
    from Distill.research_oracle import assess
    build(a.out,_data,assess,v1.load_tokenizer(a.tokenizer),json.loads(a.v1_manifest.read_text()),
          a.target_positions,a.pilot,prior,a.dev_per_profile,a.max_roots_per_profile)


if __name__=="__main__":
    main()
