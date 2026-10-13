#!/usr/bin/env python3
"""
run_design_matrix.py -- run a documented subset of the design matrix (../benchdesign/matrix.py,
frozen copy in scripts/orig/) with the UNMODIFIED generator and post-processor of the design job.

    run_design_matrix.py <outDir> <block> [--list]

Blocks (run in this order by the queue):
    cheap     every point of P4-*, P3-*, P1-sheared, P1-frame, P1-passes, P2-trap          (403 runs)
    p1n16     P1-patch, 16 cells per side, every ratio/angle/coupling                      (147)
    p1n8      P1-patch,  8 cells per side                                                 (147)
    p2L12     P2-annulus levels 1 and 2, every ratio, both variants and couplings          (56)
    p1n32     P1-patch, 32 cells per side                                                 (147)
    p2L4      P2-annulus level 4                                                          (28)
    p1n64     P1-patch, 64 cells per side: ratio 35 at every angle, ratio 1000 at 45 deg   (24)
    p2L8      P2-annulus level 8: ratio 35, variant A both couplings, variant B directional (3)
Serial, one core, niceness 10 (the queue runner applies it).  Skips cases that already have result.json.
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'orig'))
import matrix as M                       # noqa: E402

CHEAP = ('P4-ratio', 'P4-kc', 'P4-kkk', 'P4-indep', 'P3-grid', 'P3-sweep', 'P3-variants', 'P3-sheared',
         'P1-sheared', 'P1-frame', 'P1-passes', 'P2-trap')


def select(block):
    out = []
    for p in M.points():
        i, t, a = p['id'], p['tag'], p['args']
        if block == 'cheap':
            ok = t in CHEAP
        elif block in ('p1n8', 'p1n16', 'p1n32'):
            ok = t == 'P1-patch' and a['nxa'] == int(block[3:])
        elif block == 'p1n64':
            r = round(a['k1'] / a['k2']) if t == 'P1-patch' else 0
            ok = t == 'P1-patch' and a['nxa'] == 64 and (r == 35 or (r == 1000 and a['theta'] == 45))
        elif block == 'p2L12':
            ok = t == 'P2-annulus' and a['nra'] in (10, 20)
        elif block == 'p2L4':
            ok = t == 'P2-annulus' and a['nra'] == 40
        elif block == 'p2L8':
            ok = (t == 'P2-annulus' and a['nra'] == 80 and '_r35_' in i
                  and (i.startswith('p2A') or a['coupling'] == 'directional'))
        else:
            sys.exit('unknown block ' + block)
        if ok:
            out.append(p)
    return out


if __name__ == '__main__':
    out, block = sys.argv[1], sys.argv[2]
    pts = select(block)
    if '--list' in sys.argv:
        for p in pts:
            print(p['id'])
        print(len(pts), 'points')
        sys.exit(0)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, f'runlog_{block}.txt'), 'a')
    for p in pts:
        case = os.path.join(out, p['id'])
        if os.path.exists(os.path.join(case, 'result.json')):
            continue
        if p['args'].get('coupling') == 'conormal' and not os.path.exists(p['args']['conormal_lib']):
            print(p['id'], 'skipped: co-normal library missing', file=log, flush=True)
            continue
        subprocess.run(['rm', '-rf', case])
        args = [f'{k}={v}' for k, v in p['args'].items()]
        subprocess.run([sys.executable, os.path.join(HERE, 'orig', 'gencase.py'), 'new', p['problem'], case] + args,
                       check=True, stdout=subprocess.DEVNULL)
        t0 = time.time()
        r = subprocess.run([os.path.join(case, 'Allrun')], capture_output=True, text=True)
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:].replace('\n', ' ')
        print(f"{time.strftime('%H:%M:%S')} {p['id']:34s} rc={r.returncode} wall={time.time() - t0:6.1f}s {last}",
              file=log, flush=True)
