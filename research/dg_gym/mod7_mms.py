"""Opt-in MMS follow-on for the existing MOD7 affine and bijective GRU gyms.

Original pilot sources and historical receipts are untouched. Both controls
and MMS use the same explicitly exposed GRU write, initialized from the
original cell. No test or calibration data enters the training bank.
"""
import argparse
import hashlib
import importlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from datagoblin.training_gym.mms import MechanisticMemory, file_sha256


class ExposedGRUCell(nn.Module):
    """PyTorch GRUCell equations with an explicit shared recurrent affine."""
    def __init__(self, source):
        super().__init__()
        if not source.bias:
            raise ValueError('MMS MOD7 adapter expects a biased GRUCell')
        self.weight_ih = source.weight_ih
        self.bias_ih = source.bias_ih
        # Creating the wrapper does not advance the experiment RNG.
        with torch.random.fork_rng(devices=[]):
            self.recurrent = nn.Linear(source.hidden_size, 3*source.hidden_size)
        self.recurrent.weight = source.weight_hh
        self.recurrent.bias = source.bias_hh

    def forward(self, x, h):
        ir, iz, inn = F.linear(x, self.weight_ih, self.bias_ih).chunk(3, -1)
        hr, hz, hn = self.recurrent(h).chunk(3, -1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r*hn)
        return n + z*(h-n)


def keys_for(r, ops):
    # Base-49 integer key, length encoded via the leading nonzero digit.
    k = r.long()+1
    for ab in ops.transpose(0,1):
        k = 49*k+7*ab[:,0]+ab[:,1]
    return k


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    pilot = importlib.import_module('research.dg_gym.mod7_'+('cpu_pilot' if args.gym=='affine' else 'bijective_pilot'))
    torch.manual_seed(args.seed)
    net = pilot.Core()
    net.cell = ExposedGRUCell(net.cell)
    opt = torch.optim.AdamW(net.parameters(), lr=.003)
    stream = torch.Generator().manual_seed(10000+args.seed)
    memory = (MechanisticMemory(net.cell.recurrent, mib=args.mms_mib,
        mode=args.mms_mode, rank=args.rank, topk=args.topk, ratio=args.ratio,
        refresh=args.refresh, max_age=args.max_age, map_file=args.mms_map,
        map_sha=file_sha256(args.mms_map) if args.mms_map else '') if args.mms_mib else None)
    visits = count = 0
    h = hashlib.sha256()
    wall, cpu = time.perf_counter(), time.process_time()
    for i in range(args.updates):
        if args.cpu_seconds is not None and time.process_time()-cpu >= args.cpu_seconds:
            break
        length = 4 if args.gym=='affine' else (1 if i<args.updates//4 else 2 if i<args.updates//2 else 4)
        r, ops, path = pilot.sample(stream,96,length)
        h.update(keys_for(r,ops).numpy().tobytes())
        opt.zero_grad(set_to_none=True)
        context = (memory.capture(keys_for(r,ops),i,length//2,length)
                   if memory is not None else nullcontext())
        with context:
            hs = net.states(r,ops)
            loss = F.cross_entropy(net.out(hs[:,-1]),path[:,-1])
            if args.concept:
                loss = loss+.5*F.cross_entropy(net.probe(hs).reshape(-1,7),path.reshape(-1))
            loss.backward()
        opt.step()
        count += 1
        visits += 96*length
    wall_used, cpu_used = time.perf_counter()-wall, time.process_time()-cpu
    net.eval()
    curves = []
    with torch.no_grad():
        for length in (6,12,32):
            r,ops,path = pilot.sample(torch.Generator().manual_seed(82000+length),512,length)
            logits = net.out(net.states(r,ops)[:,-1])
            correct = int((logits.argmax(-1)==path[:,-1]).sum())
            heuristic=max(int((ops[:,-1,1]==path[:,-1]).sum()),
                          int(torch.bincount(path[:,-1],minlength=7).max()))
            curves.append(dict(length=length,correct=correct,n=len(r),heuristic=heuristic,
                               competent=correct>=max(256,heuristic+52),
                               ce=float(F.cross_entropy(logits,path[:,-1]))))
    args.out.mkdir(parents=True,exist_ok=False)
    torch.save(net.state_dict(),args.out/'model.pt')
    result=dict(status='PUBLIC_DEVELOPMENT_ONLY',gym=args.gym,seed=args.seed,
        parameters=sum(p.numel() for p in net.parameters()),completed_updates=count,
        root_draws=count*96,token_block_visits=visits,training_stream_sha256=h.hexdigest(),
        process_cpu_seconds=cpu_used,wall_seconds=wall_used,threads=args.threads,
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        memory=None if memory is None else memory.receipt(),curves=curves,
        mechanism_claim=False,peak_host_ram=None)
    (args.out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gym',choices=('affine','bijective'),default='bijective')
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=1200)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--cpu-seconds',type=float)
    p.add_argument('--concept',action='store_true')
    p.add_argument('--mms-mib',type=float,default=0)
    p.add_argument('--mms-mode',choices=('observe','dense','bpd','sham'),default='bpd')
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--topk',type=int,default=4)
    p.add_argument('--ratio',type=float,default=.2)
    p.add_argument('--refresh',type=int,default=128)
    p.add_argument('--max-age',type=int,default=32768)
    p.add_argument('--mms-map',default='')
    args=p.parse_args()
    if args.updates<1 or args.threads<1 or args.mms_mib<0 or (args.mms_map and not args.mms_mib):
        p.error('invalid MMS or training budget')
    if args.cpu_seconds is not None and args.cpu_seconds<=0:
        p.error('CPU budget must be positive')
    run(args)
