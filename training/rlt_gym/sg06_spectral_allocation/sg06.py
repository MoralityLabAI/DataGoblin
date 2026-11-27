"""SG-06: CPU allocation of verified dual and expansion actions."""
import copy
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from datagoblin.training_gym.spectral_dual import DualSearchSession, coordinate_minimizer, verify_dual_trace
from datagoblin.training_gym.spectral_routing import oracle, score

REGIMES = ('persistent', 'rotating', 'mixed')
ARMS = ('expand', 'root_dual', 'all_dual', 'heuristic', 'random', 'mlp', 'recurrent', 'shuffled')
CAP, PROPOSALS, DECISIONS = 2048, 24, 80
FEATURES = 12


def task(seed, regime, d=8, t=6):
    r = random.Random(seed)
    common = r.sample(range(d), 2)
    x, u = [r.randrange(-2, 3)], []
    for _ in range(t):
        x.append(x[-1] + r.randrange(1, 3))
        rotating = r.sample(range(d), 2)
        hot = common if regime == 'persistent' else rotating if regime == 'rotating' else [common[0], rotating[0]]
        u.append([r.randrange(-2, 6) + (r.randrange(9, 14) if j in hot else 0) for j in range(d)])
    return dict(d=d, T=t, bits=[r.randrange(1, 4) for _ in range(d)], x=x,
                k=[r.randrange(2, min(d, 4) + 1) for _ in range(t)], u=u, **{'lambda': r.randrange(1, 4)})


def initial(q):
    d, t = q['d'], q['T']
    def pick(i, fn):
        return sum(1 << j for j in sorted(range(d), key=lambda j: (-fn(j), j))[:q['k'][i]])
    pool = set(sorted(range(d), key=lambda j: (-sum(row[j] for row in q['u']) + q['lambda'] * q['bits'][j]
                                              * (q['x'][-1] - q['x'][0]), j))[:max(q['k'])])
    a, suffix = [0] * t, 0
    for i in range(t - 1, -1, -1):
        a[i] = pick(i, lambda j: q['u'][i][j] - (0 if suffix & (1 << j) else q['lambda'] * q['bits'][j]
                                                * (q['x'][i + 1] - q['x'][0])))
        suffix |= a[i]
    candidates = [a, [pick(i, lambda j: q['u'][i][j]) for i in range(t)],
                  [pick(i, lambda j: q['u'][i][j] + 100000 * (j in pool)) for i in range(t)]]
    return max(candidates, key=lambda a: score(q, a)['objective'])


def best(s):
    return min(s.frontier(), key=lambda a: (-a[2], a[0], a[1]))


def eta(s, t, mask):
    e = [0] * t
    for i in range(t - 1, -1, -1):
        e[i] = coordinate_minimizer(s.task, t, mask, e, i)
    return e


def observe(s, t, mask, ub, tried, steps):
    node = s.live[t, mask]
    runner = max([s.incumbent, *[n.upper for k, n in s.live.items() if k != (t, mask)]])
    return [t / s.t, mask.bit_count() / s.d, (ub - s.incumbent) / 64, (ub - runner) / 64,
            (s.weak[t] - (ub - node.value)) / 128, math.log1p(len(s.live)) / 5,
            len(s.actions[t - 1]) / 128, s.lam / 3, s.transitions / CAP,
            s.proposals / PROPOSALS, steps / DECISIONS, float((t, mask) in tried)]


def feasible(s, t, mask, tried):
    return s.transitions + len(s.actions[t - 1]) <= s.cap, (t, mask) not in tried and s.proposals < s.proposal_cap


def step(s, t, mask, action, tried):
    if action:
        s.propose(t, mask, eta(s, t, mask))
        tried.add((t, mask))
    else:
        s.advance(t, mask)


def potential(s):
    g = [max(0, n.upper - s.incumbent) for n in s.live.values()]
    return max(g, default=0) + .15 * sum(g) / max(1, len(g))


def teach(q):
    s = DualSearchSession(q, initial(q), transition_cap=CAP, proposal_cap=PROPOSALS)
    tried, xs, ys, eligible = set(), [], [], []
    for i in range(DECISIONS):
        if not s.live:
            break
        t, mask, ub = best(s)
        ex, du = feasible(s, t, mask, tried)
        if not ex and not du:
            break
        x = observe(s, t, mask, ub, tried, i)
        if ex and du:
            before = potential(s)
            a, b = copy.deepcopy(s), copy.deepcopy(s)
            a.advance(t, mask)
            b.propose(t, mask, eta(b, t, mask))
            ea = (before - potential(a) + a.incumbent - s.incumbent) / len(s.actions[t - 1])
            eb = (before - potential(b) + b.incumbent - s.incumbent) / (3 * t)
            action = int(eb > max(0, ea))
        else:
            action = int(du)
        xs.append(x)
        ys.append(action)
        eligible.append(ex and du)
        step(s, t, mask, action, tried)
    return xs, ys, eligible


class Policy(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.layer = nn.Linear(FEATURES, 12)
        self.memory = nn.Linear(12, 12, bias=False) if recurrent else None
        self.out = nn.Linear(12, 1)

    def forward(self, x, h=None):
        z = self.layer(x)
        if self.memory is not None:
            z = z + self.memory(torch.zeros_like(z) if h is None else h)
        h = torch.tanh(z)
        return self.out(h).squeeze(-1), h


def train(sequences, kind):
    torch.manual_seed(6016)
    model = Policy(kind != 'mlp')
    opt = torch.optim.Adam(model.parameters(), lr=.012)
    length = max(len(x) for x, _, _ in sequences)
    x = torch.zeros(len(sequences), length, FEATURES)
    y = torch.zeros(len(sequences), length)
    mask = torch.zeros_like(y, dtype=torch.bool)
    r = random.Random(6017)
    for j, (xx, yy, mm) in enumerate(sequences):
        labels = list(yy)
        if kind == 'shuffled':
            indices = [i for i, ok in enumerate(mm) if ok]
            shuffled = [labels[i] for i in indices]
            r.shuffle(shuffled)
            for i, v in zip(indices, shuffled, strict=True):
                labels[i] = v
        if xx:
            x[j, :len(xx)] = torch.tensor(xx)
            y[j, :len(xx)] = torch.tensor(labels)
            mask[j, :len(xx)] = torch.tensor(mm)
    for _ in range(120):
        h, out = None, []
        for i in range(length):
            z, h = model(x[:, i], h)
            out.append(z)
        logits = torch.stack(out, 1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval(), float(loss.detach()), sum(p.numel() for p in model.parameters())


@torch.no_grad()
def episode(q, arm, models, seed, cpu_limit):
    started = time.process_time_ns()
    s = DualSearchSession(q, initial(q), transition_cap=CAP, proposal_cap=PROPOSALS)
    tried, h, r = set(), None, random.Random(seed)
    coord, steps, dual_ns, expand_ns, policy_ns, reason = 0, 0, 0, 0, 0, 'decision_cap'
    for i in range(DECISIONS):
        if not s.live:
            reason = 'certified'
            break
        if cpu_limit and time.process_time_ns() - started >= cpu_limit:
            reason = 'cpu_deadline'
            break
        tick = time.process_time_ns()
        t, mask, ub = best(s)
        ex, du = feasible(s, t, mask, tried)
        if not ex and not du:
            reason = 'caps'
            break
        x = observe(s, t, mask, ub, tried, i)
        if arm in models:
            z, h = models[arm](torch.tensor(x), h)
            action = int(z.item() >= 0)
        elif arm == 'all_dual':
            action = 1
        elif arm == 'root_dual':
            action = int(t == s.t)
        elif arm == 'heuristic':
            action = int(t > 1 and len(s.actions[t - 1]) >= 4 * t and x[2] > .05)
        elif arm == 'random':
            action = r.randrange(2)
        else:
            action = 0
        action = int(du) if not ex or not du else action
        policy_ns += time.process_time_ns() - tick
        tick = time.process_time_ns()
        step(s, t, mask, action, tried)
        spent = time.process_time_ns() - tick
        if action:
            dual_ns += spent
            coord += t
        else:
            expand_ns += spent
        steps += 1
    receipt = s.receipt()
    elapsed = time.process_time_ns() - started
    if not receipt['gap']:
        reason = 'certified'
    return dict(**receipt, cpu_ns=elapsed, dual_ns=dual_ns, expand_ns=expand_ns, policy_ns=policy_ns,
                coordinate_updates=coord, decisions=steps, stop=reason,
                overshoot_ns=max(0, elapsed - cpu_limit) if cpu_limit else 0)


def campaign():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    clock = time.process_time_ns()
    sequences = [teach(task(860000 + 1000 * j + i, reg)) for j, reg in enumerate(REGIMES) for i in range(16)]
    model_info, models = {}, {}
    for kind in ('mlp', 'recurrent', 'shuffled'):
        models[kind], loss, params = train(sequences, kind)
        model_info[kind] = dict(loss=loss, parameters=params)
    training_ns = time.process_time_ns() - clock
    rows = []
    for j, reg in enumerate(REGIMES):
        for i in range(16):
            seed = 870000 + 1000 * j + i
            q = task(seed, reg, 8 if i < 12 else 9, 6 if i < 12 else 7)
            optimum = oracle(q)['objective']
            init = initial(q)
            arms = {}
            for budget in ('work_caps', 'cpu_20ms'):
                order = list(ARMS)
                random.Random(seed + (budget == 'cpu_20ms')).shuffle(order)
                arms[budget] = {}
                for arm in order:
                    z = episode(q, arm, models, seed, 20_000_000 if budget == 'cpu_20ms' else 0)
                    assert z['objective'] <= optimum <= z['upper'] and z['transitions'] <= CAP and z['proposals'] <= PROPOSALS
                    replay = verify_dual_trace(q, init, z['events'], transition_cap=CAP, proposal_cap=PROPOSALS)
                    assert all(z[k] == v for k, v in replay.items())
                    z['regret'] = optimum - z['objective']
                    arms[budget][arm] = z
            rows.append(dict(seed=seed, regime=reg, task=q, optimum=optimum, initial=init, arms=arms))
    summary = {}
    for budget in ('work_caps', 'cpu_20ms'):
        summary[budget] = {}
        for arm in ARMS:
            zz = [r['arms'][budget][arm] for r in rows]
            summary[budget][arm] = {k: statistics.mean(z[k] for z in zz) for k in
                                    ('gap', 'regret', 'transitions', 'proposals', 'coordinate_updates', 'cpu_ns',
                                     'dual_ns', 'expand_ns', 'policy_ns', 'overshoot_ns')}
            summary[budget][arm].update(certified=sum(z['gap'] == 0 for z in zz),
                                        optimal=sum(z['regret'] == 0 for z in zz),
                                        cpu_deadlines=sum(z['stop'] == 'cpu_deadline' for z in zz),
                                        max_overshoot_ns=max(z['overshoot_ns'] for z in zz))
    return dict(version='SG-06/v0.1', evidence='offline synthetic development, CPU single host', torch=torch.__version__,
                training_roots=len(sequences), teacher_eligible=sum(sum(m) for _, _, m in sequences),
                teacher_dual=sum(sum(y for y, ok in zip(yy, mm, strict=True) if ok) for _, yy, mm in sequences),
                training_cpu_ns=training_ns, models=model_info, eval_roots=len(rows),
                replayed_certificates=len(rows) * len(ARMS) * 2, summary=summary, rows=rows)


if __name__ == '__main__':
    z = campaign()
    z['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(__file__).with_name('results.json').write_text(json.dumps(z, indent=2) + '\n')
    print(json.dumps({k: v for k, v in z.items() if k != 'rows'}, indent=2))
