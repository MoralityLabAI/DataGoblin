"""Generate relying-party fixtures and, optionally, run native Nova adversarial smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from datagoblin.zk_replay import Job, Plan, execute
from datagoblin.zk_schedule import native_bundle


def write(path: Path, value: object) -> str:
    raw = json.dumps(value, separators=(',', ':'), sort_keys=True).encode('ascii')
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(n: int, depth: int) -> Plan:
    jobs = tuple(Job(f'j{i}', tuple(f'j{p}' for p in range(max(0,i-2),i)), (i%7,-2),
                     ((1,2),(-1,1)), (0,1), 1+i%depth) for i in range(n))
    return Plan('tenant', f'smoke-{n}-{depth}', 'public-dev', jobs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nodes', type=int, choices=(1,3,8,32), default=3)
    p.add_argument('--depth', type=int, choices=(1,2,4,8,16,32), default=4)
    p.add_argument('--native', type=Path)
    p.add_argument('--timeout', type=float, default=900)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = fixture(args.nodes,args.depth)
    statement, advice = native_bundle(plan)
    assert [int(v) for v in statement['zn'][3:]] == [v for r in execute(plan) for v in r['trace'][-1]]
    s, a, proof = [args.out / f for f in ('statement.json','advice.json','proof.bin')]
    pin = write(s,statement)
    write(a,advice)
    (args.out / 'statement.sha256').write_text(pin+'\n',encoding='ascii')
    print(f'prepared nodes={args.nodes} constraints={len(statement["constraints"])} pin={pin}')
    if args.native is None:
        return
    exe = str(args.native.resolve(strict=True))

    def run(label: str, ok: bool, *words: str) -> None:
        r = subprocess.run([exe,*words],capture_output=True,text=True,timeout=args.timeout,check=False)
        print(label,r.returncode,r.stdout.strip(),r.stderr.strip()[-1000:])
        if ok:
            assert r.returncode == 0 and (label != 'verify' or 'accepted' in r.stdout), label
        else:
            assert r.returncode != 0 and 'accepted' not in r.stdout, label

    run('prove',True,'prove',str(s),pin,str(a),str(proof))
    run('verify',True,'verify',str(s),pin,str(proof))
    run('wrong-pin',False,'verify',str(s),'0'*64,str(proof))
    changed = dict(statement, zn=[*statement['zn'][:-1],str(int(statement['zn'][-1])+1)])
    bad = args.out / 'wrong-output.json'
    bad_pin = write(bad,changed)
    run('wrong-output',False,'verify',str(bad),bad_pin,str(proof))
    raw = proof.read_bytes()
    corrupt = args.out / 'corrupt.bin'
    corrupt.write_bytes(raw+b'\0')
    run('trailing-bytes',False,'verify',str(s),pin,str(corrupt))
    corrupt.write_bytes(raw[:len(raw)//2])
    run('truncated',False,'verify',str(s),pin,str(corrupt))
    altered = bytearray(raw); altered[len(altered)//2] ^= 1
    corrupt.write_bytes(altered)
    run('mutated-proof',False,'verify',str(s),pin,str(corrupt))
    print('native smoke passed')


if __name__ == '__main__':
    main()
