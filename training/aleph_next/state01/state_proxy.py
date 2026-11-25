"""Exploratory CPU proxy for ALEPH-STATE-01. Run with Python 3 and torch."""
import json
import time
import torch
from torch import nn
import torch.nn.functional as F

SEEDS = (137, 163, 191)
ARMS = ('terminal', 'state', 'shuffled')
STEPS, WORDS, WIDTH, TEST_WORDS = 240, 16, 24, 128
LENGTHS = (12, 24, 48)

def words(g, n, length, reset):
    ops = torch.randint(0, 3, (n, length), generator=g)
    if reset:
        rows = torch.arange(n)
        cols = torch.randint(1, length - 1, (n,), generator=g)
        ops[rows, cols] = torch.randint(3, 7, (n,), generator=g)
    return ops

def expand(ops):
    n, length = ops.shape
    x = torch.arange(4).repeat(n)
    ops = ops.repeat_interleave(4, dim=0)
    states = []
    for t in range(length):
        op = ops[:, t]
        x = torch.where(op < 3, (x + op + 1) % 4, op - 3)
        states.append(x)
    return ops, torch.stack(states, 1)

class Executor(nn.Module):
    def __init__(self):
        super().__init__()
        self.init = nn.Linear(4, WIDTH)
        self.cell = nn.GRUCell(7, WIDTH)
        self.readout = nn.Linear(WIDTH, 4)

    def forward(self, ops):
        x = F.one_hot(torch.arange(4).repeat(ops.shape[0] // 4), 4).float()
        h = self.init(x)
        logits = []
        for op in F.one_hot(ops, 7).float().unbind(1):
            h = self.cell(op, h)
            logits.append(self.readout(h))
        return torch.stack(logits, 1)

@torch.no_grad()
def evaluate(model):
    model.eval()
    g = torch.Generator().manual_seed(2026112501)
    result = {}
    for length in LENGTHS:
        for reset in (False, True):
            ops, target = expand(words(g, TEST_WORDS, length, reset))
            pred = model(ops).argmax(-1)
            correct = pred.eq(target)
            last = correct[:, -1].reshape(-1, 4)
            result[f'{length}/'+('reset' if reset else 'no_reset')] = {
                'answer_correct': int(last.sum()), 'answer_total': last.numel(),
                'orbit_correct': int(last.all(1).sum()), 'orbit_total': TEST_WORDS,
                'trace_correct': int(correct.sum()), 'trace_total': correct.numel(),
                'full_trace_orbit_correct': int(correct.reshape(-1, 4, length).all(2).all(1).sum())}
    return result

def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {'torch': torch.__version__, 'seeds': SEEDS, 'steps': STEPS,
              'words_per_step': WORDS, 'width': WIDTH, 'test_words_per_cell': TEST_WORDS,
              'train_lengths': [4, 8], 'test_lengths': LENGTHS, 'runs': []}
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            model = Executor()
            optim = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
            g = torch.Generator().manual_seed(seed + 10000)
            shuffle = torch.Generator().manual_seed(seed + 20000)
            positions = 0
            started = time.process_time()
            for step in range(STEPS):
                length = 4 if step % 2 == 0 else 8
                w = torch.cat([words(g, WORDS // 2, length, False),
                               words(g, WORDS // 2, length, True)])
                ops, target = expand(w)
                positions += ops.numel()
                logits = model(ops)
                loss = F.cross_entropy(logits[:, -1], target[:, -1])
                if arm != 'terminal':
                    aux = target[:, :-1]
                    if arm == 'shuffled':
                        aux = torch.stack([v[torch.randperm(len(v), generator=shuffle)]
                                           for v in aux.unbind(1)], 1)
                    loss = loss + F.cross_entropy(logits[:, :-1].reshape(-1, 4), aux.reshape(-1))
                optim.zero_grad()
                loss.backward()
                optim.step()
            result['runs'].append({'seed': seed, 'arm': arm, 'positions': positions,
                                   'parameters': sum(p.numel() for p in model.parameters()),
                                   'cpu_seconds': round(time.process_time()-started, 3),
                                   'held': evaluate(model)})
    return result

if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
