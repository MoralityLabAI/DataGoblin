"""MOD7-FB-01: matched-exposure closed-loop steering."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F

SEEDS = (11, 23, 37)
ARMS = ('task', 'concept', 'static', 'adaptive', 'shuffled_feedback')
LENGTHS = (6, 32, 128, 512)


def sample(g, n, length):
    return torch.randint(7, (n,), generator=g), torch.randint(42, (n, length), generator=g)


def ids(r, ops):
    z = r.clone()
    for op in ops.T:
        z = 42*z + op
    return z


def exact(r, ops):
    z = r.clone()
    out = []
    for op in ops.T:
        z = ((op//7+1)*z + op%7)%7
        out.append(z)
    return torch.stack(out, 1)


def sample_clean(g, n, length, exclude):
    r, ops = sample(g, n, length)
    rejected = 0
    if exclude is not None:
        mask = torch.isin(ids(r, ops), exclude)
        while bool(mask.any()):
            k = int(mask.sum())
            rejected += k
            r[mask], ops[mask] = sample(g, k, length)
            mask = torch.isin(ids(r, ops), exclude)
    return r, ops, rejected


def donors(z):
    n = len(z)
    i = torch.arange(n)
    d = torch.full_like(i, -1)
    w = torch.full_like(i, -1)
    for shift in range(1, n):
        k = (i+shift)%n
        use = (d<0)&(z[k]!=z)
        d[use] = k[use]
        if bool((d>=0).all()):
            break
    if bool((d<0).any()):
        raise ValueError('donor unavailable')
    for shift in range(1, n):
        k = (i+shift)%n
        use = (w<0)&(z[k]!=z)&(z[k]!=z[d])
        w[use] = k[use]
        if bool((w>=0).all()):
            break
    if bool((w<0).any()):
        raise ValueError('wrong donor unavailable')
    return d, w


def rollout(r, ops, m, edit=None):
    p = F.one_hot(r, 7).to(m.dtype)
    out = []
    for j, op in enumerate(ops.T):
        p = torch.bmm(p.unsqueeze(1), m[op]).squeeze(1)
        if edit is not None and j == edit[0]:
            p = (1-edit[1])*p + edit[1]*edit[2]
        out.append(p)
    return torch.stack(out, 1)


def mass(p, z):
    return p.gather(1, z[:, None]).squeeze(1)


def nll(p, z):
    return -mass(p, z).clamp_min(1e-12).log().mean()


@torch.no_grad()
def audit(m, r, ops, sites):
    z = exact(r, ops)
    h = rollout(r, ops, m)
    rows = []
    for j in sites:
        d, w = donors(z[:, j])
        cf = exact(z[d, j], ops[:, j+1:])[:, -1]
        assert bool((cf!=z[:, -1]).all())
        base = mass(h[:, -1], cf)
        for a in (0.5, 1.0):
            p = rollout(r, ops, m, (j, a, h[d, j]))[:, -1]
            q = rollout(r, ops, m, (j, a, h[w, j]))[:, -1]
            good, wrong = mass(p, cf), mass(q, cf)
            s = float((good-torch.maximum(base, wrong)).mean())
            rows.append(dict(site=j, dose=a, specificity=s, deficit=1-s/a,
                             donor_mass=float(good.mean()), identity_mass=float(base.mean()),
                             wrong_mass=float(wrong.mean()), donor_correct=int((p.argmax(1)==cf).sum()),
                             identity_correct=int((h[:, -1].argmax(1)==cf).sum()),
                             wrong_correct=int((q.argmax(1)==cf).sum())))
    return rows


def train(seed, arm, feedback):
    g = torch.Generator().manual_seed(seed)
    logits = torch.nn.Parameter(torch.randn((42, 7, 7), generator=g)*0.1)
    opt = torch.optim.Adam([logits], lr=.05)
    stream = torch.Generator().manual_seed(10000+seed)
    windows = []
    train_visits = audit_visits = rejected = 0
    audit_seconds = 0.
    start, cpu = time.monotonic(), time.process_time()
    selected = None
    for i in range(600):
        length = 1 if i<150 else 2 if i<300 else 4
        if i%50==0:
            selected = None
            if length>1:
                t = time.monotonic()
                fr, fo, _ = feedback[length]
                scores = audit(logits.detach().softmax(-1), fr, fo, range(length-1))
                audit_visits += len(fr)*length*(1+2*len(scores))
                best = max(range(len(scores)), key=lambda k, scores=scores: (scores[k]['deficit'], -k))
                index = (i//50)%len(scores)
                if arm=='adaptive':
                    index = best
                elif arm=='shuffled_feedback':
                    sg = torch.Generator().manual_seed(70000+seed+i//50)
                    index = int(torch.randperm(len(scores), generator=sg)[best])
                selected = scores[index]
                windows.append(dict(update=i, length=length, selected=index, greedy=best, scores=scores))
                audit_seconds += time.monotonic()-t
        r, ops, nr = sample_clean(stream, 96, length, feedback[length][2] if length>1 else None)
        rejected += nr
        z = exact(r, ops)
        m = logits.softmax(-1)
        h = rollout(r, ops, m)
        loss = nll(h[:, -1], z[:, -1])
        train_visits += len(r)*length
        if selected is not None:
            j, a = selected['site'], selected['dose']
            d, _ = donors(z[:, j])
            cf = exact(z[d, j], ops[:, j+1:])[:, -1]
            p = rollout(r, ops, m, (j, a, h[d, j]))[:, -1]
            aux = (1-a)*nll(p, z[:, -1])+a*nll(p, cf)
            loss = loss+(0.5 if arm in ARMS[2:] else 0.)*aux
            if arm=='concept':
                loss = loss+0.5*torch.stack([nll(h[:, k], z[:, k]) for k in range(length-1)]).mean()
            train_visits += len(r)*length
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return logits.detach(), dict(train_seconds=time.monotonic()-start-audit_seconds,
                                 audit_seconds=audit_seconds, cpu_seconds=time.process_time()-cpu,
                                 train_visits=train_visits, audit_visits=audit_visits,
                                 rejected_draws=rejected, accepted_draws=57600, updates=600, windows=windows)


@torch.no_grad()
def evaluate(logits, data):
    m = logits.double().softmax(-1)
    i = torch.arange(42)
    rows = ((i[:, None]//7+1)*torch.arange(7)[None, :]+i[:, None]%7)%7
    row_correct = int((m.argmax(-1)==rows).sum())
    eps = float((1-m.gather(2, rows[..., None]).squeeze(-1)).max())
    curves = []
    for length, (r, ops, z) in data.items():
        p = rollout(r, ops, m)[:, -1]
        target = z[:, -1]
        v = mass(p, target)
        assert float(v.min()) >= (1-eps)**length-1e-10
        curves.append(dict(length=length, correct=int((p.argmax(1)==target).sum()), n=len(r),
                           mass=float(v.mean()), min_mass=float(v.min())))
    competent = row_correct==294 and curves[0]['correct']>=461
    patch = audit(m, *data[32][:2], (15,))[1] if competent else None
    return dict(row_correct=row_correct, worst_row_tv=eps, competent=competent, curves=curves, patch32=patch)


def main(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    feedback, data = {}, {}
    for k in (2, 4):
        r, o = sample(torch.Generator().manual_seed(61000+k), 96, k)
        feedback[k] = (r, o, ids(r, o).unique())
    for k in LENGTHS:
        r, o = sample(torch.Generator().manual_seed(62000+k), 512, k)
        data[k] = (r, o, exact(r, o))
    fits = []
    for seed in SEEDS:
        for arm in ARMS:
            logits, stats = train(seed, arm, feedback)
            raw = logits.numpy().astype('<f4').tobytes()
            (out/f'{seed}_{arm}.f32').write_bytes(raw)
            t = time.monotonic()
            row = dict(seed=seed, arm=arm, parameters=logits.numel(), **stats, **evaluate(logits, data),
                       checkpoint_sha256=hashlib.sha256(raw).hexdigest(), eval_seconds=time.monotonic()-t)
            fits.append(row)
            print(seed, arm, row['curves'][-1]['mass'], row['competent'], flush=True)
    lookup = {(r['seed'], r['arm']): r['curves'][-1]['mass'] for r in fits}
    gains = {c: [lookup[s, 'adaptive']-lookup[s, c] for s in SEEDS] for c in ARMS[2::2]}
    passed = all(min(v)>0 and sum(v)/len(v)>=.01 for v in gains.values())
    passed = passed and all(r['competent'] for r in fits if r['arm']=='adaptive')
    result = dict(status='PUBLIC_DEVELOPMENT_ONLY',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, platform=platform.platform(), cpu_threads=1, fits=fits,
                  paired_mass_gains=gains, screen_passed=passed, peak_ram=None, gpu_rates=None)
    (out/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
