"""MOD7-STATE-01 fixed-exposure public-development pilot."""
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F


SEEDS = (11, 23, 37)
ARMS = ('task', 'concept', 'shuffled')
LENGTHS = (6, 32, 128, 512)


def sample(g, n, length):
    r = torch.randint(7, (n,), generator=g)
    a = torch.randint(1, 7, (n, length), generator=g)
    b = torch.randint(7, (n, length), generator=g)
    z = r.clone()
    states = []
    for j in range(length):
        z = (a[:, j] * z + b[:, j]) % 7
        states.append(z)
    return r, (a - 1) * 7 + b, torch.stack(states, 1)


def step(p, op, matrices):
    return torch.bmm(p.unsqueeze(1), matrices[op]).squeeze(1)


def rollout(p, ops, matrices):
    for j in range(ops.shape[1]):
        p = step(p, ops[:, j], matrices)
    return p


def train(seed, arm):
    g = torch.Generator().manual_seed(seed)
    logits = torch.nn.Parameter(torch.randn((42, 7, 7), generator=g) * 0.1)
    opt = torch.optim.Adam([logits], lr=0.05)
    stream = torch.Generator().manual_seed(10000 + seed)
    wall, cpu = time.monotonic(), time.process_time()
    for i in range(600):
        length = 1 if i < 150 else 2 if i < 300 else 4
        r, ops, states = sample(stream, 96, length)
        p = F.one_hot(r, 7).float()
        matrices = logits.softmax(-1)
        auxiliary = []
        for j in range(length):
            p = step(p, ops[:, j], matrices)
            if j < length - 1 and arm != 'task':
                labels = states[:, j] if arm == 'concept' else states[:, j].roll(1)
                auxiliary.append(-p.gather(1, labels[:, None]).clamp_min(1e-12).log().mean())
        loss = -p.gather(1, states[:, -1:]).clamp_min(1e-12).log().mean()
        if auxiliary:
            loss = loss + 0.5 * torch.stack(auxiliary).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return logits.detach(), time.monotonic() - wall, time.process_time() - cpu


@torch.no_grad()
def evaluate(logits, data):
    matrices = logits.double().softmax(-1)
    ids = torch.arange(42)
    rows = ((ids[:, None] // 7 + 1) * torch.arange(7)[None, :] + ids[:, None] % 7) % 7
    hard = matrices.argmax(-1)
    epsilon = float((1 - matrices.gather(2, rows[..., None]).squeeze(-1)).max())
    audit = int((hard == rows).sum())
    curves = []
    for length, (r, ops, states) in data.items():
        start = time.monotonic()
        p = rollout(F.one_hot(r, 7).double(), ops, matrices)
        z = r.clone()
        for j in range(length):
            z = hard[ops[:, j], z]
        target = states[:, -1]
        mass = p.gather(1, target[:, None]).squeeze(1)
        lower = (1 - epsilon) ** length
        assert float(mass.min()) >= lower - 1e-10
        assert float((1 - mass).max()) <= min(1, length * epsilon) + 1e-10
        curves.append(dict(length=length, n=len(r), soft_correct=int((p.argmax(-1) == target).sum()),
                           hard_correct=int((z == target).sum()), mean_mass=float(mass.mean()),
                           min_mass=float(mass.min()), path_mass_lower=lower,
                           tv_upper=min(1, length * epsilon), eval_wall_seconds=time.monotonic() - start))
    competent = audit == 294 and curves[0]['soft_correct'] >= 461
    patches = None
    if competent:
        r, ops, states = data[32]
        z = states[:, 15]
        p = rollout(F.one_hot(r, 7).double(), ops[:, :16], matrices)
        donors, shams = [], []
        for i in range(len(z)):
            donor = next(k % len(z) for k in range(i + 1, i + len(z)) if z[k % len(z)] != z[i])
            sham = next(k % len(z) for k in range(donor + 1, donor + len(z))
                        if z[k % len(z)] != z[donor] and z[k % len(z)] != z[i])
            donors.append(donor)
            shams.append(sham)
        donor_ids = torch.tensor(donors)
        target = z[donor_ids].clone()
        for j in range(16, 32):
            target = rows[ops[:, j], target]
        assert bool((target != states[:, -1]).all())
        edits = {'identity': p, 'donor': p[donor_ids], 'wrong_donor': p[torch.tensor(shams)],
                 'oracle': F.one_hot(z[donor_ids], 7).double()}
        patches = {}
        for name, edit in edits.items():
            q = rollout(edit, ops[:, 16:], matrices)
            patches[name] = dict(correct=int((q.argmax(-1) == target).sum()),
                                 mean_mass=float(q.gather(1, target[:, None]).mean()))
    return dict(row_correct=audit, max_row_tv=epsilon, competent=competent, curves=curves, patches=patches)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = {k: sample(torch.Generator().manual_seed(57000 + k), 512, k) for k in LENGTHS}
    fits = []
    for seed in SEEDS:
        for arm in ARMS:
            logits, wall, cpu = train(seed, arm)
            row = dict(seed=seed, arm=arm, parameters=logits.numel(), updates=600, draws=57600, visits=158400,
                       train_wall_seconds=wall, train_cpu_seconds=cpu,
                       checkpoint_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(), **evaluate(logits, data))
            fits.append(row)
            print(json.dumps(row), flush=True)
    receipt = dict(status='PUBLIC_DEVELOPMENT_ONLY', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   torch=torch.__version__, platform=platform.platform(), cpu_threads=1, budget='fixed exposure',
                   per_fit_peak_ram=None, gpu_rates=None, fits=fits)
    Path('mod7_state_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
