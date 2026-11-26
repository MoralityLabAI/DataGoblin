"""MOD7-FB-02: optimizer-aware steering with a common visit-credit cap."""
import argparse
import copy
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import torch
from mod7_feedback import donors, exact, ids, mass, nll, rollout, sample, sample_clean

SEEDS = (11, 23, 37)
ARMS = ('adaptive', 'static_shadow', 'shuffled_shadow', 'static_reinvest', 'task_reinvest')
LENGTHS = (6, 32, 128, 512)
CAP = 1010112


def dataset(seed, n, length):
    r, o = sample(torch.Generator().manual_seed(seed), n, length)
    return r, o, exact(r, o)


def digest(p):
    return hashlib.sha256(p.detach().numpy().astype('<f4').tobytes()).hexdigest()


def update(p, opt, batch, site, weight):
    r, o, z = batch
    m = p.softmax(-1)
    h = rollout(r, o, m)
    loss = nll(h[:, -1], z[:, -1])
    if site is not None:
        d, _ = donors(z[:, site])
        cf = exact(z[d, site], o[:, site+1:])[:, -1]
        patch = rollout(r, o, m, (site, 1., h[d, site]))[:, -1]
        loss = loss+weight*nll(patch, cf)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return len(r)*o.shape[1]*(1+int(site is not None))


@torch.no_grad()
def monitor(p, data):
    r, o, z = data
    h = rollout(r, o, p.double().softmax(-1))[:, -1]
    return float(mass(h, z[:, -1]).mean())


@torch.no_grad()
def specificity(p, data):
    r, o, z = data
    m = p.double().softmax(-1)
    h = rollout(r, o, m)
    scores = []
    for j in range(3):
        d, w = donors(z[:, j])
        cf = exact(z[d, j], o[:, j+1:])[:, -1]
        b = mass(h[:, -1], cf)
        x = mass(rollout(r, o, m, (j, 1., h[d, j]))[:, -1], cf)
        y = mass(rollout(r, o, m, (j, 1., h[w, j]))[:, -1], cf)
        scores.append(float((x-torch.maximum(b, y)).mean()))
    return scores


def lookahead(p, opt, batch, cohort):
    before = digest(p)
    state = copy.deepcopy(opt.state_dict())
    scores = []
    visits = 0
    for action in (-1, 0, 1, 2):
        q = torch.nn.Parameter(p.detach().clone())
        qo = torch.optim.Adam([q], lr=.05)
        qo.load_state_dict(copy.deepcopy(state))
        visits += update(q, qo, batch, max(action, 0), .5 if action >= 0 else 0.)
        scores.append(monitor(q.detach(), cohort))
    assert digest(p) == before
    for key, value in state['state'][0].items():
        assert torch.equal(opt.state_dict()['state'][0][key], value)
    return scores[0], [s-scores[0] for s in scores[1:]], visits


def train(seed, arm, audit, cohort):
    g = torch.Generator().manual_seed(seed)
    p = torch.nn.Parameter(torch.randn(42, 7, 7, generator=g)*.1)
    opt = torch.optim.Adam([p], lr=.05)
    stream = torch.Generator().manual_seed(10000+seed)
    excluded = ids(*audit[:2]).unique()
    tf = lf = af = mf = rejected = steps = credits = 0
    records = []
    audit_s = clone_s = 0.
    start, cpu = time.monotonic(), time.process_time()
    shadow = arm.endswith('shadow') or arm == 'adaptive'
    selected = None
    while True:
        k = 1 if steps < 150 else 2 if steps < 300 else 4
        window = (steps-300)//50
        site = None if k == 1 else 0 if k == 2 else window%3
        if arm == 'task_reinvest' and k == 4:
            site = None
        next_credit = 3*96*k*(1+int(site is not None))
        if shadow and steps == 600:
            break
        if not shadow and credits+next_credit > CAP:
            break
        r, o, n = sample_clean(stream, 96, k, excluded if k == 4 else None)
        batch = r, o, exact(r, o)
        rejected += n
        if shadow and k == 4:
            if (steps-300)%50 == 0:
                t = time.monotonic()
                spec = specificity(p.detach(), audit)
                audit_s += time.monotonic()-t
                af += 64*4*7
                t = time.monotonic()
                base, gains, v = lookahead(p, opt, batch, cohort)
                clone_s += time.monotonic()-t
                lf += v
                mf += 4*64*24
                best = max(gains)
                greedy = min(i for i, s in enumerate(gains) if best-s <= 1e-9)
                if arm == 'adaptive':
                    selected = greedy
                elif arm == 'shuffled_shadow':
                    sg = torch.Generator().manual_seed(71000+seed+window)
                    selected = int(torch.randperm(3, generator=sg)[greedy])
                else:
                    selected = window%3
                records.append(dict(step=steps, site=selected, greedy=greedy, specificity=spec,
                                    task_clone_score=base, gains=gains))
            site = selected
        tf += update(p, opt, batch, site, .5 if site is not None else 0.)
        steps += 1
        credits = 3*(tf+lf)+af+mf
        assert credits <= CAP
    assert (credits == CAP if shadow else CAP-credits < next_credit)
    return p.detach(), dict(updates=steps, committed_draws=steps*96, clone_draws=len(records)*4*96,
                            rejected_draws=rejected, train_forward=tf, clone_forward=lf, audit_forward=af,
                            monitor_forward=mf, backward_visit_proxy=tf+lf, credits=credits, unused=CAP-credits,
                            wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu,
                            audit_seconds=audit_s, clone_monitor_seconds=clone_s, windows=records)


@torch.no_grad()
def evaluate(p, data):
    m = p.double().softmax(-1)
    i = torch.arange(42)
    rows = ((i[:, None]//7+1)*torch.arange(7)[None, :]+i[:, None]%7)%7
    correct = int((m.argmax(-1)==rows).sum())
    eps = float((1-m.gather(2, rows[..., None]).squeeze(-1)).max())
    curves = []
    for k, (r, o, z) in data.items():
        h = rollout(r, o, m)[:, -1]
        target = z[:, -1]
        v = mass(h, target)
        assert float(v.min()) >= (1-eps)**k-1e-10
        curves.append(dict(length=k, correct=int((h.argmax(1)==target).sum()), n=len(r), mass=float(v.mean())))
    return dict(row_correct=correct, worst_row_tv=eps, curves=curves,
                competent=correct==294 and curves[0]['correct']>=461)


def main(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    audit = dataset(63004, 64, 4)
    cohort = dataset(63024, 64, 24)
    data = {k: dataset(64000+k, 512, k) for k in LENGTHS}
    fits = []
    for seed in SEEDS:
        for arm in ARMS:
            p, stats = train(seed, arm, audit, cohort)
            raw = p.numpy().astype('<f4').tobytes()
            (out/f'{seed}_{arm}.f32').write_bytes(raw)
            t = time.monotonic()
            row = dict(seed=seed, arm=arm, parameters=p.numel(), checkpoint_sha256=digest(p), **stats,
                       **evaluate(p, data), eval_seconds=time.monotonic()-t)
            fits.append(row)
            print(seed, arm, row['curves'][-1]['mass'], row['updates'], row['credits'], flush=True)
    lookup = {(r['seed'], r['arm']): r['curves'][-1]['mass'] for r in fits}
    gains = {a: [lookup[s, 'adaptive']-lookup[s, a] for s in SEEDS] for a in ARMS[1:]}
    screen = all(min(gains[a])>0 and sum(gains[a])/3>=.01 for a in ARMS[1:4])
    screen = screen and all(r['competent'] for r in fits if r['arm']=='adaptive')
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (Path(__file__), Path(__file__).with_name('mod7_feedback.py'))}
    receipt = dict(status='PUBLIC_DEVELOPMENT_ONLY', credit_cap=CAP, fits=fits, paired_gains=gains,
                   screen_passed=screen, cpu_threads=1, torch=torch.__version__, platform=platform.platform(),
                   process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   per_fit_peak_ram=None, gpu_rates=None, sources=sources)
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
