"""Generate relying-party fixtures and, optionally, run native Nova adversarial smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from datagoblin import zk_frontier, zk_journal
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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--frontier', action='store_true')
    mode.add_argument('--journal', action='store_true')
    p.add_argument('--timeout', type=float, default=900)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = fixture(args.nodes,args.depth)
    if args.journal:
        statement, advice = zk_journal.native_bundle(plan)
        claim = tuple(map(int, statement['zn']))
        assert zk_journal.claimed_outputs(plan, claim) == tuple(r['trace'][-1] for r in execute(plan))
        assert zk_journal.authorized_statement(plan, claim) == statement
    elif args.frontier:
        statement, advice = zk_frontier.native_bundle(plan)
        assert zk_frontier.terminal(plan, tuple(map(int, statement['zn']))) == execute(plan)[-1]['trace'][-1]
    else:
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
    altered = bytearray(raw)
    altered[len(altered)//2] ^= 1
    corrupt.write_bytes(altered)
    run('mutated-proof',False,'verify',str(s),pin,str(corrupt))
    if args.journal:
        root = hashlib.sha256(b'authorized synthetic root').hexdigest()
        outputs = zk_journal.claimed_outputs(plan, claim)
        items = [{'job': plan.jobs[0].name, 'root': root, 'split': 'TRAIN', 'payload': outputs[0]}]
        roots = {plan.jobs[0].name: root}
        exe_pin = hashlib.sha256(Path(exe).read_bytes()).hexdigest()
        receipt = zk_journal.verify_and_certify(plan, claim, raw, Path(exe), exe_pin, roots, {root}, items)
        write(args.out / 'corpus-receipt.json', receipt)
        print(f'journal admission accepted count={receipt["manifest"]["count"]}')
        if args.nodes > 1:
            idx = len(claim) - zk_journal.limb_count(plan)
            old = claim[idx] & 255
            new = 1 if old == 0 else 0
            changed_claim = (*claim[:idx], claim[idx]-old+new, *claim[idx+1:])
            changed_outputs = zk_journal.claimed_outputs(plan, changed_claim)
            assert changed_outputs[-1] == outputs[-1] and changed_outputs[0] != outputs[0]
            changed_items = [{**items[0], 'payload': changed_outputs[0]}]
            try:
                zk_journal.verify_and_certify(
                    plan, changed_claim, raw, Path(exe), exe_pin, roots, {root}, changed_items,
                )
            except ValueError as exc:
                assert str(exc) == 'native verifier rejected claim'
                print('retired-output substitution rejected')
            else:
                raise AssertionError('retired-output substitution accepted')
    print('native smoke passed')


if __name__ == '__main__':
    main()
