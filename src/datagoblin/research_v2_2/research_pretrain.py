"""Build a fresh DG research pretraining candidate; never launch a trainer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from datagoblin.research_v2_2 import closure_proof, deficit_cover, native_probe, skill_transfer
from datagoblin.research_v2_2._exact_engine import binary_key
from datagoblin.research_v2_2.exact_research import ExactResearch

UPSTREAM = "7fa19c90372b13d71676e6fef13819d2a18b9be3"
SOURCE = "1a73ecc4db75857973f9472ca86bc435af48b433"
TOKENIZER = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
DELIMITER = "\nAnswer\n"
VERSION = "dg-research-pretrain-v1"
FAMILIES = ("adaptive_evidence", "robust_discrimination", "research_portfolio",
            "proof_obligations", "paired_causal_gate", "research_memory")
SEEDS = {"train": 2026851000, "development": 2026861000}
DIFFICULTIES = ("foundation", "standard", "frontier")
INSTRUCTIONS = {
    "adaptive_evidence": "Maximize expected net reward. Tests reveal a deterministic bit, cost budget and utility.",
    "robust_discrimination": ("Minimize experiment cost. Require dependencies "
                              "and distance at least 2e+1 for every pair."),
    "research_portfolio": ("Maximize minimum total scenario gain with CPU, RAM, "
                           "permission, group and dependency limits."),
    "proof_obligations": "Maximize conditional checked-claim value within audit budget; respect claim prerequisites.",
    "paired_causal_gate": ("Compute equal-root active-minus-sham mean, positive roots "
                           "and visit/dose matching for each arm."),
    "research_memory": "Compare checked-cache policies in the stipulated constant-cost cyclic model.",
}


def canonical(x):
    return json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode("ascii")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def identity(kind, d):
    return digest(dict(kind=kind, data=d))


def semantic(kind, d):
    if kind == "adaptive_evidence":
        return hashlib.sha256(binary_key(d)[0]).hexdigest()
    return identity(kind, d)


def old_roots(gen):
    """Known public native inventories; private/global exclusion is a later gate."""
    old = set()

    def add(kind, seed, n, difficulties):
        for i in range(n):
            d = gen(kind, seed, i, difficulties[i%len(difficulties)])
            old.add(semantic(kind, d))

    for seed, n in ((2026112801, 48), (2026112802, 16), (2026112803, 32),
                    (2026770000, 128), (2026780000, 32), (2026790000, 64)):
        add("adaptive_evidence", seed, n, ("standard", "frontier"))
    add("adaptive_evidence", 2026800000, 64, ("foundation",))
    for k in FAMILIES[1:5]:
        for seed, n in ((2026811000, 24), (2026812000, 32)):
            add(k, seed, n, ("standard", "frontier"))
    for j, diff in enumerate(DIFFICULTIES):
        add("robust_discrimination", 2026821000+1000*j, 24, (diff,))
    add("robust_discrimination", 2026831000, 24, ("frontier",))
    add("robust_discrimination", 2026841000, 24, ("frontier",))
    # Structured C/D robust worlds use the same native-shaped schema.
    from datagoblin.research_v2_2.dependency_compact import bundled
    for base, count in ((2026831000, 12), (2026841000, 16)):
        for j in (1, 2):
            for i in range(count):
                old.add(semantic("robust_discrimination", closure_proof.structured(base+1000*j+i,j==2)))
    for i in range(16):
        old.add(semantic("robust_discrimination", bundled(2026844000+i)))
    return old


def fraction(x):
    x = Fraction(x)
    return f"{x.numerator}/{x.denominator}"


def compact_world(kind, d):
    if kind != "paired_causal_gate":
        return d
    fields = ("root","active","sham","active_visits","sham_visits","active_dose","sham_dose")
    return dict(root_columns=list(fields), min_positive=d["min_positive"], min_mean=d["min_mean"],
                arms=[dict(id=a["id"],roots=[[r[k] for k in fields] for r in a["roots"]]) for a in d["arms"]])


def solve_native(kind, d):
    if kind == "adaptive_evidence":
        r = ExactResearch().solve(d)
        return dict(policy=r["policy"],value=r["value"]),dict(method="checked_integer_bellman")
    if kind == "robust_discrimination":
        a,_,p = closure_proof.solve(d,"forced",65536)
        if p is None:
            raise ValueError("missing exact proof")
        receipt = closure_proof.verify(d,p)
        return a,dict(method="independent_branch_proof",proof_nodes=receipt["nodes"],proof_sha256=digest(p))
    if kind == "paired_causal_gate":
        return skill_transfer.paired_answer(d),dict(method="equal_root_rational_arithmetic")
    p = skill_transfer.Problem(kind,d)
    best, val = 0,p.inspect(0)[2]
    for mask in range(1 << p.n):
        r = p.inspect(mask)
        if r is not None and r[2] > val:
            best,val = mask,r[2]
    return p.answer(best,True),dict(method="exhaustive_dependency_closed_subsets")


def corrupt(kind, answer):
    bad = copy.deepcopy(answer)
    if kind == "adaptive_evidence":
        bad["value"] = fraction(Fraction(bad["value"])+1)
        why = "The stated utility differs from the exact replay of the submitted policy."
    elif kind == "robust_discrimination":
        if bad["status"] == "plan":
            bad["cost"] += 1
            why = "The claimed panel cost differs from the selected experiments' total."
        else:
            bad = dict(status="plan",experiments=[],cost=0)
            why = "The empty panel cannot meet the positive pair-separation requirement."
    elif kind == "research_portfolio":
        bad["worst_gain"] += 1
        why = "The claimed worst-case gain differs from the minimum scenario sum."
    elif kind == "proof_obligations":
        bad["conditional_value"] += 1
        why = "The conditional claim value differs from the dependency-checked closure."
    else:
        a = bad["arms"][0]
        a["mean_delta"] = fraction(Fraction(a["mean_delta"])+1)
        why = "The claimed mean differs from equal-root active-minus-sham arithmetic."
    return bad,why


def baseline(kind, d):
    if kind == "adaptive_evidence":
        policy,_ = native_probe.tree(d,"one_step")
        from datagoblin.research_v2_2.exact_research import policy_value
        return dict(policy=policy,value=fraction(policy_value(d,policy))),"Receding one-step search"
    if kind == "robust_discrimination":
        answer,_ = deficit_cover.solve(d,"greedy",4096)
        return answer,"Dependency-closure deficit greedy; unresolved is not infeasibility"
    if kind == "paired_causal_gate":
        return skill_transfer.paired_answer(d,True),"Pooled observation means, instead of equal-root means"
    p = skill_transfer.Problem(kind,d)
    best,val = 0,p.inspect(0)[2]
    for mask in p.closures:
        r = p.inspect(mask)
        if r is not None and r[2] > val:
            best,val = mask,r[2]
    return p.answer(best,True),"Best single dependency closure, including empty baseline"


def grade_summary(g):
    return {k:g[k] for k in ("feasible","optimal","certificate_consistent","candidate_value","oracle_value","gap")}


def native_views(kind,d,assess):
    case = dict(kind=kind,data=d)
    answer,witness = solve_native(kind,d)
    grade = assess(case,answer)
    if not grade["optimal"] or not grade["feasible"] or not grade["certificate_consistent"]:
        raise ValueError("independent native oracle rejected generated target")
    witness = dict(**witness,checker="Distill.research_oracle.assess",grade=grade_summary(grade))
    world = INSTRUCTIONS[kind]+"\nWorld: "+canonical(compact_world(kind,d))
    bad,why = corrupt(kind,answer)
    g = assess(case,bad)
    if g["optimal"]:
        raise ValueError("fault injection did not fail")
    control,label = baseline(kind,d)
    cg = assess(case,control) if control is not None else None
    # Reference and proposal stay distinct: state actual finite-model grades.
    comparison = dict(control=label,control_grade=grade_summary(cg) if cg else "unresolved",
                      optimal_answer=answer,scope="finite stipulated model")
    if kind == "paired_causal_gate":
        comparison["eligible_changed"] = control["eligible"] != answer["eligible"]
    return [
        ("decision",world+"\nReturn an optimal decision certificate.",answer,witness),
        ("repair",world+"\nAudit this proposed certificate and repair it: "+canonical(bad),
         dict(verdict="reject",reason=why,repair=answer),
         dict(**witness,corrupted_grade=grade_summary(g))),
        ("matched_control",world+"\nCompare the following baseline against the exact optimum. Baseline: "
         +canonical(dict(method=label,answer=control)),comparison,
         dict(**witness,control_grade=grade_summary(cg) if cg else None)),
    ]


def memory_world(seed,i):
    rng = random.Random(seed*1000003+i)
    if i%3 == 0:
        m = rng.randrange(4,25)
        return dict(kind="marginal_rank",m=m,k=rng.randrange(m),classes=2)
    d = rng.randrange(3,49)
    return dict(kind="cyclic_cache",roots=d,cycles=rng.randrange(2,15),
                capacity=rng.randrange(1,d),rent=rng.randrange(2,81),
                hit=rng.randrange(0,2),verify=rng.randrange(1,161))


def simulation(d,policy):
    n,r,c = d["roots"],d["cycles"],d["capacity"]
    s,h,v = d["rent"],d["hit"],d["verify"]
    seen,cache = set(),[]
    hits,checks,solves = 0,0,0
    for t in range(n*r):
        k = t%n
        if k in cache:
            hits += 1
            cache.remove(k)
            cache.append(k)
        else:
            solves += 1
            if policy == "eager" or (policy in ("second","freeze") and k in seen):
                if len(cache) < c or policy != "freeze":
                    if len(cache) == c:
                        cache.pop(0)
                    cache.append(k)
                    checks += 1
        seen.add(k)
    return dict(hits=hits,checks=checks,solves=solves,cost=hits*h+checks*v+solves*s)


def memory_views(d):
    prompt = "Finite research-memory model: "+canonical(d)
    if d["kind"] == "marginal_rank":
        m,k = d["m"],d["k"]
        rank = 2*sum(math.comb(m,j) for j in range(k+1))
        domain = 2**(m+1)
        # Independent Pascal recurrence for the rank.
        row = [1]
        for _ in range(m):
            row = [a+b for a,b in zip([0,*row],[*row,0],strict=True)]
        if 2*sum(row[:k+1]) != rank or 2*sum(row)!=domain:
            raise ValueError("rank replay")
        a = dict(domain=domain,rank=rank,kernel_dimension=domain-rank,
                 meaning="linear dimensions of unnormalized class-conditioned binary marginals")
        witness = dict(method="binomial_and_pascal_replay",scope="finite analytic model")
        return [
            ("decision",prompt+"\nCompute rank and invisible dimension of all marginals through order k.",a,witness),
            ("repair",prompt+"\nAudit the claim that equal order-k marginals certify exact joint identity.",
             dict(verdict="reject",kernel_dimension=domain-rank,
                  reason="A Walsh parity character of degree above k is invisible to these marginals."),witness),
            ("matched_control",prompt+"\nCompare order k with full-order marginal information.",
             dict(order_k_rank=rank,full_rank=domain,missing=domain-rank,
                  scope="linear coefficients, not Shannon information"),witness),
        ]
    n,r,c = d["roots"],d["cycles"],d["capacity"]
    s,h,v = d["rent"],d["hit"],d["verify"]
    total = n*r
    values = {a:simulation(d,a) for a in ("cold","eager","second","freeze")}
    formula = dict(hits=(r-2)*c,checks=c,solves=total-(r-2)*c,
                   cost=(total-(r-2)*c)*s+(r-2)*c*h+c*v)
    if (values["freeze"] != formula or values["eager"]["hits"] != 0 or values["second"]["hits"] != 0
            or values["eager"]["checks"] != total or values["second"]["checks"] != (r-1)*n):
        raise ValueError("cyclic ledger replay")
    saving = c*((r-2)*(s-h)-v)
    if values["cold"]["cost"]-formula["cost"] != saving:
        raise ValueError("payback arithmetic")
    witness = dict(method="explicit_request_simulation_vs_closed_form",scope="constant costs; exact cyclic overflow")
    return [
        ("decision",prompt+"\nCompute cold, eager LRU, delayed LRU and freeze hit/check/solve and cost ledgers.",
         values,witness),
        ("repair",prompt+"\nAudit the claim that delaying LRU admission always prevents repeated verification.",
         dict(verdict="reject",delayed=values["second"],
              reason="A complete cycle evicts the earlier key before its next use."),
         witness),
        ("matched_control",prompt+"\nDerive freeze's net saving versus cold and decide whether verification pays back.",
         dict(saving=saving,profitable=saving>0,break_even="(R-2)*(s-h)>v",
              scope="excludes key and monitor overhead"),witness),
    ]


def load_tokenizer(path):
    if path is None:
        return None
    import tokenizers
    if tokenizers.__version__ != "0.22.2" or file_hash(path)!=TOKENIZER:
        raise ValueError("wrong frozen tokenizer")
    tok = tokenizers.Tokenizer.from_file(str(path))
    tok.encode_special_tokens = True
    if tok.get_vocab_size()!=49152 or 0 in tok.encode("<|endoftext|>",add_special_tokens=False).ids:
        raise ValueError("tokenizer geometry or literal control token")
    return tok


def pack(tok,prompt,target):
    prefix = prompt+DELIMITER
    text = prefix+target+"\n"
    obj = tok.encode(text,add_special_tokens=False)
    if len(obj.ids)>2048:
        return None
    if not obj.ids or any(not 0<=x<49152 for x in obj.ids) or tok.decode(
            obj.ids,skip_special_tokens=False)!=text:
        raise ValueError("tokenizer round trip")
    mask = [lo>=len(prefix) and hi>lo for lo,hi in obj.offsets]
    if not any(mask) or mask[0] or not all(mask[mask.index(True):]):
        raise ValueError("response boundary")
    return dict(token_ids=obj.ids,target_mask=mask,response_start=mask.index(True))


def build(out,gen,assess,counts,tok=None):
    out.mkdir(parents=True,exist_ok=False)
    forbidden = old_roots(gen)
    seen = set(forbidden)
    exact_texts = set()
    manifest = dict(schema=VERSION,source_commit=SOURCE,distill_commit=UPSTREAM,
                    source_kind="fresh_project_generated_synthetic",known_excluded_roots=len(forbidden),
                    max_views_per_root=3,max_presentations_per_root=4,
                    tokenizer_sha256=TOKENIZER if tok else None,training_ready=False,
                    release_scope="candidate; independent steward and global exclusion/rights audit pending",
                    splits={},sources={"native":"Distill/research_cases.py","checker":"Distill/research_oracle.py",
                    "memory":"RV22-S/U/X analytic models"},production_launch_authorized=False)
    for split,n in zip(("train","development"),counts,strict=True):
        folder = out/split
        folder.mkdir()
        roots,stats = [],defaultdict(Counter)
        rejected = Counter()
        with (folder/"records.jsonl").open("w",encoding="ascii") as raw, \
             (folder/"packed.jsonl").open("w",encoding="ascii") as packed:
            for fi,kind in enumerate(FAMILIES):
                accepted,ordinal = 0,0
                while accepted<n:
                    if ordinal>=max(128,n*16):
                        raise ValueError("generator exhausted")
                    seed = SEEDS[split]+fi*1000
                    difficulty = DIFFICULTIES[ordinal%3]
                    d = (memory_world(seed,ordinal) if kind=="research_memory"
                         else gen(kind,seed,ordinal,difficulty))
                    origin = dict(kind="fresh_synthetic",family=kind,seed=seed,ordinal=ordinal,
                                  difficulty=difficulty,source_commit=SOURCE if kind=="research_memory" else UPSTREAM)
                    ordinal += 1
                    root = semantic(kind,d)
                    if root in seen:
                        rejected["root_overlap"] += 1
                        continue
                    views = memory_views(d) if kind=="research_memory" else native_views(kind,d,assess)
                    pending = []
                    for view,prompt,target,witness in views:
                        target = canonical(target)
                        transcript = prompt+DELIMITER+target+"\n"
                        th = hashlib.sha256(transcript.encode("ascii")).hexdigest()
                        if th in exact_texts:
                            raise ValueError("duplicate rendered transcript")
                        item = pack(tok,prompt,target) if tok else None
                        if tok and item is None:
                            rejected["over_context_views"] += 1
                            continue
                        record = dict(split=split,skill="custom_research",lane="custom_research",family=kind,
                                      semantic_key=root,record_id=digest([root,view]),view=view,
                                      objective="checked_response",prompt=prompt,target=target,delimiter=DELIMITER,
                                      origin=origin,witness=witness,record_sha256=th,
                                      custody="candidate",training_admission=False)
                        pending.append((record,item,th))
                    if not pending:
                        rejected["empty_roots"] += 1
                        continue
                    seen.add(root)
                    roots.append(root)
                    accepted += 1
                    stats[kind]["roots"] += 1
                    for record,item,th in pending:
                        exact_texts.add(th)
                        raw.write(canonical(record)+"\n")
                        s = stats[kind]
                        s["records"] += 1
                        s["prompt_bytes"] += len(record["prompt"].encode("ascii"))
                        s["target_bytes"] += len(record["target"].encode("ascii"))
                        s["transcript_bytes"] += len((record["prompt"]+DELIMITER+record["target"]+"\n").encode("ascii"))
                        s["views_"+record["view"]] += 1
                        if item is not None:
                            s["positions"] += len(item["token_ids"])
                            s["loss_tokens"] += sum(item["target_mask"])
                            s["max_context"] = max(s["max_context"],len(item["token_ids"]))
                            packed.write(canonical(dict(**item,root=root,family=kind,lane="custom_research",split=split,
                                objective="checked_response",presentation=1,record_id=record["record_id"],
                                receipt_sha256=digest(record),training_admission=False))+"\n")
        if tok is None:
            (folder/"packed.jsonl").unlink()
        (folder/"roots.json").write_text(json.dumps(sorted(roots),indent=2)+"\n")
        manifest["splits"][split] = dict(roots=len(roots),records=sum(s["records"] for s in stats.values()),
            families={k:dict(v) for k,v in stats.items()},rejected=dict(rejected),
            files={p.name:dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(folder.iterdir())})
    if len(seen)-len(forbidden) != sum(x["roots"] for x in manifest["splits"].values()):
        raise ValueError("split ledger")
    manifest["manifest_id"] = digest(manifest)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    # Separate train and development releases, each with the common aggregate manifest.
    for split in ("train","development"):
        (out/split/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("DG_RESEARCH_DATASET "+canonical(manifest),flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--distill-checkout",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--train-per-family",type=int,default=512)
    p.add_argument("--dev-per-family",type=int,default=64)
    p.add_argument("--tokenizer",type=Path)
    a = p.parse_args()
    if not 1<=a.train_per_family<=4096 or not 1<=a.dev_per_family<=512:
        raise ValueError("bounded root count")
    path = a.distill_checkout.resolve()
    sha = subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
    dirty = subprocess.check_output(["git","-C",str(path),"status","--porcelain"],text=True).strip()
    if sha!=UPSTREAM or dirty:
        raise ValueError("clean pinned upstream required")
    sys.path.insert(0,str(path))
    from Distill.research_cases import _data
    from Distill.research_oracle import assess
    build(a.out,_data,assess,(a.train_per_family,a.dev_per_family),load_tokenizer(a.tokenizer))


if __name__=="__main__":
    main()
