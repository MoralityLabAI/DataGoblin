"""Read-only validation of a pinned EvalMath AIRIS-07 training-candidate bundle.

No training admission, network access, evaluator/development input, or model execution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

UPSTREAM = "3ef134f0d880a81ad56592487d951ac2287237d5"
SCHEMA = "evalmath-airis07-candidate-v1"
EXPECTED_SHA = "988fa25090d00918705f1affae0916dd7faf51a06d58588c3b98e591a845723d"
SCHEDULES = ((0,1,0),) + tuple(a for a in product(range(2), repeat=3) if a != (0,1,0))
OBS = tuple(product(range(2),repeat=2))
RAW = tuple(product(range(2),repeat=3))
ROW_KEYS = {"schema","root_id","group_id","presentation","split","training_ready",
            "input","provenance","teacher"}
INPUT_KEYS = {"drift_resample_probability","driver_flip_probability","history",
              "next_episode_transition_cost","candidate_schedule_order"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def read_bounded(path, maximum):
    need(path.is_file() and not path.is_symlink(), "expected regular local file")
    need(path.stat().st_size <= maximum, "file too large")
    return path.read_bytes()


@lru_cache(None)
def emission(a, raw, theta):
    # Independent full raw-bit likelihood, enumerating initial hidden driver
    # and flip paths. Does not import EvalMath or its temporal-code oracle.
    total = F(0)
    for u,f,g in product(range(2),repeat=3):
        driver=(u,u^f,u^f^g)
        prev=(0,a[0],a[1])
        y=tuple((x*(theta&1))^(p*((theta>>1)&1))^b
                for x,p,b in zip(a,prev,driver))
        if y==raw:
            total+=F(1,2)*F(1 if f else 3,4)*F(1 if g else 3,4)
    return total


def predict(w,k):
    return tuple((1-F(k,4))*x+F(k,16)*sum(w) for x in w)


@lru_cache(None)
def oracle(history,k):
    w=(F(1,4),)*4
    for a,y in history:
        w=tuple(x*emission(a,y,i) for i,x in enumerate(predict(w,k)))
        z=sum(w)
        need(z>0,"impossible history")
        w=tuple(x/z for x in w)
    w=predict(w,k)
    return tuple(sum(max(w[i]*emission(a,y,i) for i in range(4)) for y in RAW)
                 for a in SCHEDULES)


def validate_row(row):
    need(type(row) is dict and set(row)==ROW_KEYS,"unexpected row fields")
    need(row["schema"]==SCHEMA and row["split"]=="train_candidate" and
         row["training_ready"] is False,"candidate-only custody required")
    v=row["presentation"]
    need(type(v) is int and v in (0,1),"invalid presentation")
    data=row["input"]
    need(type(data) is dict and set(data)==INPUT_KEYS,"unexpected input fields")
    need(data["drift_resample_probability"] in ("0/4","1/4"),"unregistered training rate")
    k=int(data["drift_resample_probability"][0])
    need(data["driver_flip_probability"]=="1/4" and
         type(data["next_episode_transition_cost"]) is int and
         data["next_episode_transition_cost"]==3,"wrong mechanism contract")
    need(data["candidate_schedule_order"]==["".join(map(str,a)) for a in SCHEDULES],
         "wrong candidate order")
    need(type(data["history"]) is list and len(data["history"])==3,"wrong history length")
    history=[]
    codes=[]
    for e in data["history"]:
        need(type(e) is dict and set(e)=={"actions","observations"},"unexpected evidence fields")
        for key in ("actions","observations"):
            need(type(e[key]) is list and len(e[key])==3 and
                 all(type(x) is int and x in (0,1) for x in e[key]),"invalid evidence bits")
        a,y=tuple(e["actions"]),tuple(e["observations"])
        need(y[0]==0,"noncanonical raw lift")
        history.append((a,y))
        codes.append(4*SCHEDULES.index(a)+OBS.index((y[0]^y[1],y[1]^y[2])))
    i,j=sorted(codes[:2])
    need(i<j and codes[:2]==([i,j] if v==0 else [j,i]),"invalid chronological pairing")
    group=digest([SCHEMA,"group",[i,j,codes[2]]])
    root=digest([SCHEMA,k,group])
    need(row["group_id"]==group and row["root_id"]==root,"identity mismatch")
    need(int(group[:8],16)%5!=0,"development group in training")
    need(row["provenance"]=={"generator":"EvalMath/gyms/AIRIS/nonstationary.py",
                              "evidence":"finite synthetic exact oracle"},"provenance mismatch")
    q=oracle(tuple(history),k)
    teacher=row["teacher"]
    need(type(teacher) is dict and set(teacher)=={"action_values","optimal_actions"},
         "unexpected supervision fields")
    need(teacher["action_values"]==[str(x) for x in q],"independent oracle mismatch")
    best=[i for i,x in enumerate(q) if x==max(q)]
    need(type(teacher["optimal_actions"]) is list and
         all(type(x) is int for x in teacher["optimal_actions"]) and
         teacher["optimal_actions"]==best,"incorrect optimal action set")
    return group,k,v,tuple(best),tuple(codes)


def validate_bundle(directory, expected_sha=EXPECTED_SHA):
    directory=Path(directory)
    m=json.loads(read_bounded(directory/"train_manifest.json",16384))
    need(set(m)=={"schema","training_ready","partition","rows","groups","file","sha256","claim_boundary"},
         "unexpected manifest fields")
    need(m["schema"]==SCHEMA and m["training_ready"] is False and
         m["partition"]=="train_candidate" and m["file"]=="train_candidates.jsonl" and
         m["claim_boundary"]=="synthetic candidate proposals; no admission or learned result",
         "wrong training manifest")
    raw=read_bounded(directory/m["file"],4*1024*1024)
    sha=sha256(raw).hexdigest()
    need(sha==m["sha256"]==expected_sha,"pinned corpus hash mismatch")
    groups=defaultdict(dict)
    n=0
    for line in raw.splitlines():
        need(len(line)<=8192,"oversized row")
        group,k,v,best,codes=validate_row(json.loads(line))
        need((k,v) not in groups[group],"duplicate presentation")
        groups[group][k,v]=(best,codes)
        n+=1
    need(n==m["rows"] and len(groups)==m["groups"],"manifest count mismatch")
    for g,rows in groups.items():
        need(set(rows)==set(product((0,1),repeat=2)),"missing paired control")
        for v in (0,1):
            need(rows[0,v][1]==rows[1,v][1],"stationary control mismatch")
        need(rows[0,0][0]==rows[0,1][0],"stationary control changes answer")
        need(not set(rows[1,0][0])&set(rows[1,1][0]),"strict drift pair lacks action separation")
    need(n>0,"empty corpus")
    return {"schema":"datagoblin-evalmath-airis07-intake-v1","upstream_commit":UPSTREAM,
            "source_sha256":sha,"candidate_rows":n,"observation_groups":len(groups),
            "independent_oracle_rows":n,"status":"CANDIDATE_ONLY","training_ready":False}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("bundle",type=Path)
    args=p.parse_args()
    print(json.dumps(validate_bundle(args.bundle),sort_keys=True,indent=2))
