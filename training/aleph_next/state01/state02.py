"""Fixed-dose exploratory finite-state transfer proxy (CPU, PyTorch)."""
import json
import time
import torch
from torch import nn
from torch.nn import functional as F

SEEDS = (211, 223, 227)
ARMS = ('terminal', 'state', 'shuffled')
STEPS, WORDS, WIDTH, N = 960, 18, 24, 128
LENGTHS = (12, 24, 48)
REGIMES = ('survival', 'partial', 'reset', 'double_partial')
# 3 permutations, 2 balanced rank-2 maps, 4 constant maps.
MAPS = torch.tensor([[1,2,3,0], [1,0,2,3], [0,3,2,1],
                     [0,2,0,2], [0,0,1,1], [0,0,0,0],
                     [1,1,1,1], [2,2,2,2], [3,3,3,3]])
PAIRS = torch.triu_indices(4, 4, 1)

def words(g, n, length, regime):
    w = torch.randint(0, 3, (n, length), generator=g)
    rows = torch.arange(n)
    if regime == 'double_partial':
        for lo, hi in ((1, length//2), (length//2, length-1)):
            cols = torch.randint(lo, hi, (n,), generator=g)
            w[rows, cols] = torch.randint(3, 5, (n,), generator=g)
    elif regime in ('partial', 'reset'):
        cols = torch.randint(1, length-1, (n,), generator=g)
        lo, hi = (3,5) if regime == 'partial' else (5,9)
        w[rows, cols] = torch.randint(lo, hi, (n,), generator=g)
    return w

def expand(w):
    ops = w.repeat_interleave(4, 0)
    s = torch.arange(4).repeat(len(w))
    trace = []
    for u in ops.unbind(1):
        s = MAPS[u, s]
        trace.append(s)
    return ops, torch.stack(trace, 1)

class Executor(nn.Module):
    def __init__(self):
        super().__init__()
        self.init = nn.Linear(4, WIDTH)
        self.cell = nn.GRUCell(len(MAPS), WIDTH)
        self.readout = nn.Linear(WIDTH, 4)

    def forward(self, ops):
        start = F.one_hot(torch.arange(4).repeat(len(ops)//4), 4).float()
        h = self.init(start)
        logits = []
        for u in F.one_hot(ops, len(MAPS)).float().unbind(1):
            h = self.cell(u, h)
            logits.append(self.readout(h))
        return torch.stack(logits, 1)

def metrics(pred, target):
    length = target.shape[1]
    p, y = (a.reshape(-1, 4, length) for a in (pred, target))
    ok = p.eq(y)
    p_eq = p[:, PAIRS[0]] == p[:, PAIRS[1]]
    y_eq = y[:, PAIRS[0]] == y[:, PAIRS[1]]
    equiv = p_eq.eq(y_eq)
    pf, yf = p[:, :, -1], y[:, :, -1]
    fibers = F.one_hot(yf, 4).sum(1)
    return {'answers': int(ok[:, :, -1].sum()), 'answer_total': len(p)*4,
            'orbits': int(ok[:, :, -1].all(1).sum()), 'orbit_total': len(p),
            'traces': int(ok.all(1).all(1).sum()),
            'terminal_partitions': int(equiv[:, :, -1].all(1).sum()),
            'trace_partitions': int(equiv.all(1).all(1).sum()),
            'suffix_bound_answers': int(fibers.max(1).values.sum()),
            'suffix_orbits': int((fibers.max(1).values == 4).sum()),
            'ranks': [int((fibers.ne(0).sum(1)==r).sum()) for r in range(1,5)]}

@torch.no_grad()
def evaluate(model, panel):
    return {key: metrics(model(ops).argmax(-1), y) for key, (ops,y) in panel.items()}

def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.equal(MAPS[0, MAPS[1]], MAPS[1, MAPS[0]])
    assert [len(x.unique()) for x in MAPS] == [4,4,4,2,2,1,1,1,1]
    g = torch.Generator().manual_seed(2026112502)
    panel = {f'{l}/{r}': expand(words(g,N,l,r)) for l in LENGTHS for r in REGIMES}
    for l in LENGTHS:
        for r, rank in zip(REGIMES[:3], (4,2,1)):
            y = panel[f'{l}/{r}'][1]
            assert metrics(y,y)['ranks'][rank-1] == N
    out = {'seeds': SEEDS, 'steps': STEPS, 'panel_seed': 2026112502,
           'torch': torch.__version__, 'train_lengths':[4,8], 'runs':[]}
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            model = Executor()
            opt = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
            g = torch.Generator().manual_seed(seed+10000)
            sg = torch.Generator().manual_seed(seed+20000)
            positions = 0
            start = time.process_time()
            for step in range(STEPS):
                length = 4 if step % 2 == 0 else 8
                ops, y = expand(torch.cat([words(g,WORDS//3,length,r) for r in REGIMES[:3]]))
                positions += ops.numel()
                z = model(ops)
                loss = F.cross_entropy(z[:,-1],y[:,-1])
                if arm != 'terminal':
                    aux = y[:,:-1]
                    if arm == 'shuffled':
                        aux = torch.stack([v[torch.randperm(len(v),generator=sg)] for v in aux.unbind(1)],1)
                    loss = loss + F.cross_entropy(z[:,:-1].reshape(-1,4),aux.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            out['runs'].append({'seed': seed, 'arm':arm, 'positions':positions,
                'parameters':sum(p.numel() for p in model.parameters()),
                'cpu_seconds':round(time.process_time()-start,3), 'held':evaluate(model,panel)})
            print(seed,arm,out['runs'][-1]['held']['48/survival']['orbits'],flush=True,file=__import__('sys').stderr)
    return out

if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
