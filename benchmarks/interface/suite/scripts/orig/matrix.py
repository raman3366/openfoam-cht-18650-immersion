#!/usr/bin/env python3
"""
matrix.py -- the full verification matrix (DESIGN.md section 4) as an explicit list.

    matrix.py list [tag]                 print the points (id, problem, tag, arguments)
    matrix.py count                      number of points per tag
    matrix.py run <outDir> [tag] [--max N] [--skip-done]
                                         generate and run the points one after another
                                         (serial, niceness 10); needs the OpenFOAM
                                         environment:  openfoam2606 -c "python3 matrix.py run ..."
    matrix.py csv <outDir>               one line per finished case -> <outDir>/matrix_results.csv

Tags name the hypothesis a block of runs decides (DESIGN.md section 6).
Nothing in this file has been run except the points listed in RUNLOG.md.
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

RATIOS = (1, 3, 10, 35, 100, 200, 1000)      # k_high / k_low; 35 = demonstration, 10-33 laminations,
                                             # 150-215 measured cylindrical cells (sources in DESIGN.md)
K_LOW = 0.8
ANGLES = (0, 15, 30, 45, 60, 75, 90)
CONORMAL_LIB = os.environ.get('CONORMAL_LIB', os.path.join(HERE, '..', 'derivation', 'coNormalInterface',
                                                           'lib', 'libcoNormalInterface.so'))


def points():
    P = []

    def add(tag, name, problem, **kw):
        P.append(dict(id=name, tag=tag, problem=problem, args=kw))

    # ------------------------------------------------------------- P1 patch test
    for r in RATIOS:
        for th in ANGLES:
            for cpl in ('scalar', 'directional', 'conormal'):
                base = dict(k1=K_LOW * r, k2=K_LOW, k3=K_LOW, theta=th, coupling=cpl, ki=1.0, A=1000, B=500)
                if cpl == 'conormal':
                    base['conormal_lib'] = CONORMAL_LIB
                # sheared strip: closed-form scheme solution, mesh independent -> one mesh
                add('P1-sheared', f'p1s_r{r}_t{th}_{cpl}', 'patch', ybc='gradient', nxa=16, nxi=16, ny=4,
                    nsteps=600, **base)
                # all-Dirichlet patch test on four meshes
                for n in (8, 16, 32, 64):
                    add('P1-patch', f'p1_r{r}_t{th}_{cpl}_n{n}', 'patch', nxa=n, nxi=n, ny=n,
                        nsteps=max(400, 25 * n * (4 if r >= 200 else 1)), **base)
    for th in (0, 30, 60):                               # frame invariance and coupling passes
        for phi in (20, 45):
            add('P1-frame', f'p1_r35_t{th}_phi{phi}', 'patch', k1=28, k2=0.8, k3=0.8, theta=th, phi=phi,
                coupling='directional', nxa=16, nxi=16, ny=16)
        add('P1-passes', f'p1_r35_t{th}_o3', 'patch', k1=28, k2=0.8, k3=0.8, theta=th, nouter=3,
            coupling='directional', nxa=16, nxi=16, ny=16)
    # ------------------------------------------------------------- P2 annulus
    for var, extra in (('A', dict(sector=90)), ('B', dict(sector=360))):
        for r in RATIOS:
            kk = dict(kr=K_LOW, kth=K_LOW, kz=K_LOW * r) if var == 'A' else dict(kr=K_LOW, kth=K_LOW * r, kz=K_LOW * r)
            for cpl in ('scalar', 'directional'):
                for lev in (1, 2, 4, 8):
                    add('P2-annulus', f'p2{var}_r{r}_{cpl}_L{lev}', 'annulus', coupling=cpl, nra=10 * lev,
                        nri=12 * lev, nth=(16 if var == 'A' else 8) * lev, nsteps=600 * lev, **kk, **extra)
    for nth in (16, 32, 64, 128):                        # symmetry-plane trap, expected first order in 1/nth
        add('P2-trap', f'p2C_r35_nth{nth}', 'annulus', kr=0.8, kth=28, kz=28, sector=90, nra=10, nri=12,
            nth=nth, coupling='directional')
    # ------------------------------------------------------------- P3 transient two-layer
    P3 = dict(ki=0.15, La=0.004, Li=0.006, rhoA=2000, cpA=1000, rhoI=1000, cpI=1500, xbc='NN', ic='step',
              Ta0=350, Ti0=300, nwrite=4)
    for cpl in ('scalar', 'directional'):
        for nouter in (1, 2, 3):
            for lev in (1, 2, 4, 8):
                for m in (1, 2, 4, 8):                   # dt = 0.4/m, end time 40 s
                    add('P3-grid', f'p3_{cpl}_o{nouter}_L{lev}_dt{m}', 'slab', k1=28, k2=0.8, k3=0.8, theta=60,
                        coupling=cpl, nouter=nouter, nxa=16 * lev, nxi=24 * lev, dt=0.4 / m, nsteps=100 * m, **P3)
    for r in (1, 10, 35, 200, 1000):
        for th in (0, 30, 60, 90):
            for cpl in ('scalar', 'directional'):
                add('P3-sweep', f'p3_r{r}_t{th}_{cpl}', 'slab', k1=K_LOW * r, k2=K_LOW, k3=K_LOW, theta=th,
                    coupling=cpl, nxa=32, nxi=48, dt=0.1, nsteps=400, **P3)
    for name, kw in (('rev', dict(order='aniso iso')), ('mode1', dict(ic='mode', mode_n=1)),
                     ('mode3', dict(ic='mode', mode_n=3)),
                     ('dd', dict(xbc='DD', ic='uniform', Ta0=300, TL=350, TR=300))):
        for m in (1, 2, 4):
            a = dict(P3); a.update(kw)
            add('P3-variants', f'p3_{name}_dt{m}', 'slab', k1=28, k2=0.8, k3=0.8, theta=60, coupling='directional',
                nxa=32, nxi=48, dt=0.4 / m, nsteps=100 * m, **a)
    for cpl in ('directional', 'conormal'):              # transient with a tangential gradient
        for m in (1, 2, 4):
            a = dict(P3); a.update(xbc='DD', ic='uniform', Ta0=300, TL=350, TR=300)
            if cpl == 'conormal':
                a['conormal_lib'] = CONORMAL_LIB
            add('P3-sheared', f'p3s_{cpl}_dt{m}', 'slab', k1=28, k2=0.8, k3=0.8, theta=30, B=2000, ny=4, H=0.002,
                coupling=cpl, nxa=32, nxi=48, dt=0.4 / m, nsteps=100 * m, **a)
    # ------------------------------------------------------------- P4 flux ratio
    P4 = dict(La=0.004, Li=0.006, rhoA=2000, cpA=1000, rhoI=1000, cpI=1500, xbc='NN', ic='step',
              Ta0=350, Ti0=300, nwrite=2, nxa=16, nxi=24, dt=0.05, nsteps=400)
    for r in RATIOS:
        for th in (0, 30, 45, 60, 90):
            for cpl in ('scalar', 'directional'):
                add('P4-ratio', f'p4_r{r}_t{th}_{cpl}', 'slab', k1=K_LOW * r, k2=K_LOW, k3=K_LOW, theta=th,
                    coupling=cpl, ki=0.15, **P4)
    for kc in (0.1, 0.4, 1.6, 5.0, 80.0):                # arbitrary scalar handed to the coupling
        add('P4-kc', f'p4_kc{kc}', 'slab', k1=28, k2=0.8, k3=0.8, theta=90, coupling='function', kc=kc, ki=0.15, **P4)
    for k in (0.5, 2.0, 50.0):                           # isotropic value entered as a vector
        for cpl in ('scalar', 'directional'):
            add('P4-kkk', f'p4_kkk{k}_{cpl}', 'slab', k1=k, k2=k, k3=k, theta=0, coupling=cpl, ki=0.15, **P4)
    for ki in (0.15, 5.0):                               # independence of neighbour, mesh, time step
        for lev, m in ((1, 1), (2, 1), (1, 4), (4, 4)):
            a = dict(P4); a.update(nxa=16 * lev, nxi=24 * lev, dt=0.05 / m, nsteps=400 * m)
            add('P4-indep', f'p4_ki{ki}_L{lev}_dt{m}', 'slab', k1=28, k2=0.8, k3=0.8, theta=90, coupling='scalar',
                ki=ki, **a)
    return P


def run(out, tag=None, nmax=None, skip_done=False):
    os.makedirs(out, exist_ok=True)
    n = 0
    for p in points():
        if tag and p['tag'] != tag:
            continue
        case = os.path.join(out, p['id'])
        if skip_done and os.path.exists(os.path.join(case, 'result.json')):
            continue
        if nmax is not None and n >= nmax:
            break
        n += 1
        if p['args'].get('coupling') == 'conormal' and not os.path.exists(p['args']['conormal_lib']):
            print(p['id'], 'skipped: co-normal library missing')
            continue
        subprocess.run(['rm', '-rf', case])
        args = [f'{k}={v}' for k, v in p['args'].items()]
        subprocess.run([sys.executable, os.path.join(HERE, 'gencase.py'), 'new', p['problem'], case] + args,
                       check=True, stdout=subprocess.DEVNULL)
        r = subprocess.run(['nice', '-n', '10', os.path.join(case, 'Allrun')], capture_output=True, text=True)
        print(p['id'], 'rc', r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:],
              flush=True)


def csv(out):
    rows = []
    for p in points():
        f = os.path.join(out, p['id'], 'result.json')
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        e, i, a = r['error'], r['interface'], r.get('audit', {})
        rows.append([p['id'], p['tag'], r['problem'], r['params']['coupling'], r['params']['nouter'],
                     r['cells']['aniso'] + r['cells']['iso'], r['params']['dt'],
                     e['exact']['all']['L2'], e['exact']['all']['Linf'],
                     e.get('scheme', {}).get('all', {}).get('Linf', ''),
                     i['recomputed_imbalance'] if r['problem'] == 'slab' else i['imbalance'],
                     a.get('ratio_received_over_released', ''), a.get('predicted_ratio_continuum', ''),
                     a.get('drift', ''), r.get('steady_change_since_previous_write', '')])
    with open(os.path.join(out, 'matrix_results.csv'), 'w') as fh:
        fh.write('id,tag,problem,coupling,nouter,cells,dt,L2_exact,Linf_exact,Linf_scheme,interface_imbalance_W_per_m,'
                 'energy_ratio,energy_ratio_predicted,energy_drift_J_per_m,steady_change_K\n')
        for row in rows:
            fh.write(','.join(str(v) for v in row) + '\n')
    print(len(rows), 'cases ->', os.path.join(out, 'matrix_results.csv'))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['list']:
        for p in points():
            if len(a) < 2 or p['tag'] == a[1]:
                print(p['id'], p['problem'], p['tag'], ' '.join(f'{k}={v}' for k, v in p['args'].items()))
    elif a[:1] == ['count']:
        c = {}
        for p in points():
            c[p['tag']] = c.get(p['tag'], 0) + 1
        for k, v in c.items():
            print(f'{k:14s} {v}')
        print(f"{'total':14s} {sum(c.values())}")
    elif a[:1] == ['run'] and len(a) >= 2:
        rest = [x for x in a[2:] if not x.startswith('--')]
        nmax = int(a[a.index('--max') + 1]) if '--max' in a else None
        rest = [x for x in rest if not (nmax is not None and x == str(nmax))]
        run(a[1], rest[0] if rest else None, nmax, '--skip-done' in a)
    elif a[:1] == ['csv'] and len(a) == 2:
        csv(a[1])
    else:
        sys.exit(__doc__)
