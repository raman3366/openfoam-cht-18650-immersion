#!/usr/bin/env python3
"""
blocks.py -- the lists of benchmark points run by runpts.py.

Two families:
  design_*   points of the design job's matrix (scripts/orig/matrix.py), same ids and
             arguments; a documented SUBSET of the 1103 points (see RESULTS.md, section 2)
  x_*        points added by the benchmark job: the corrected boundary condition, the guard,
             the curved oblique interface (ring), the audit function objects, and replicas of
             the derivation job's configurations so that every entry of predictions.json has
             a run of its own here
Each function returns [(id, problem, {args}), ...] in run order.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'orig'))
import matrix as M              # noqa: E402  (the design job's matrix, unmodified)

BENCH = os.path.dirname(HERE)
CN = os.path.join(BENCH, 'lib', 'coNormalCoupling', 'lib', 'libcoNormalCoupling.so')
FO = os.environ.get('CONORMAL_LIB', os.path.join(BENCH, '..', 'derivation', 'coNormalInterface', 'lib',
                                                 'libcoNormalInterface.so'))
K_LOW = 0.8
RATIOS = (1, 3, 10, 35, 100, 200, 1000)
ANGLES = (0, 15, 30, 45, 60, 75, 90)


def _design(pred):
    out = []
    for p in M.points():
        if pred(p):
            a = dict(p['args'])
            if a.get('coupling') == 'conormal':
                a['conormal_lib'] = FO
            out.append((p['id'], p['problem'], a))
    return out


def _ratio(p):
    a = p['args']
    return round(a['k1'] / a['k2']) if 'k1' in a else None


# ------------------------------------------------------------------ design matrix, subset


def design_cheap():
    """Every point of the one-dimensional and sheared blocks (403 points)."""
    tags = ('P1-sheared', 'P1-frame', 'P1-passes', 'P2-trap', 'P4-ratio', 'P4-kc', 'P4-kkk', 'P4-indep',
            'P3-sweep', 'P3-variants', 'P3-sheared', 'P3-grid')
    tag = {p['id']: p['tag'] for p in M.points()}
    return sorted(_design(lambda p: p['tag'] in tags), key=lambda t: tags.index(tag[t[0]]))


def design_p1n16():
    """All-Dirichlet patch test, 16 cells per side: scalar and directional at every ratio and angle,
    function-object variant at ratios 35 and 1000 (112)."""
    # the function-object variant only at ratios 35 and 1000 (the derived BC covers every point in x_p1bc)
    return _design(lambda p: p['tag'] == 'P1-patch' and p['args']['nxa'] == 16
                   and (p['args']['coupling'] != 'conormal' or _ratio(p) in (35, 1000)))


def design_p1mesh():
    """Patch test on 8 and 32 cells per side at ratio 35 (every angle) and at ratio 1000, 45 degrees,
    scalar and directional (32)."""
    return _design(lambda p: p['tag'] == 'P1-patch' and p['args']['nxa'] in (8, 32) and p['args']['coupling'] != 'conormal'
                   and (_ratio(p) == 35 or (_ratio(p) == 1000 and p['args']['theta'] == 45)))


def design_p1n64():
    """Patch test on 64 cells per side, ratio 35, 45 degrees, stock directional option (1).
    (The corrected BC on 64 cells is in x_p1bc; scalar and function-object variant were left out.)"""
    return _design(lambda p: p['tag'] == 'P1-patch' and p['args']['nxa'] == 64 and _ratio(p) == 35
                   and p['args']['theta'] == 45 and p['args']['coupling'] == 'directional')


def _p2(p, var, ratios, cpl, levels):
    i, a = p['id'], p['args']
    if p['tag'] != 'P2-annulus' or not i.startswith('p2' + var):
        return False
    r = int(i.split('_')[1][1:])
    return r in ratios and a['coupling'] in cpl and a['nra'] // 10 in levels


def design_p2():
    """Annulus: ratio 35 on levels 1, 2, 4 for both variants and couplings; ratios 1 and 1000 on
    levels 1 and 2; the remaining ratios on level 1 (38 points)."""
    both = ('scalar', 'directional')
    return _design(lambda p: _p2(p, 'A', (35,), both, (1, 2, 4)) or _p2(p, 'B', (35,), both, (1, 2, 4))
                   or _p2(p, 'A', (1, 1000), both, (1, 2)) or _p2(p, 'B', (1000,), both, (1, 2))
                   or _p2(p, 'A', (3, 10, 100, 200), both, (1,)) or _p2(p, 'B', (3, 10, 100, 200), ('directional',), (1,)))


def design_p2L8():
    """Annulus level 8, variant A, ratio 35, directional (1 point; 22528 cells)."""
    return _design(lambda p: _p2(p, 'A', (35,), ('directional',), (8,)))


# ------------------------------------------------------------------ added by the benchmark job

P1 = dict(k1=28, k2=0.8, k3=0.8, ki=1.0, A=1000, B=500)
DSLAB = dict(k1=28, k2=0.8, k3=0.8, La=0.002, Li=0.002, rhoA=2700, cpA=1050, rhoI=1000, cpI=2000, ki=0.5,
             xbc='NN', ic='step', Ta0=350, Ti0=300, nxa=10, nxi=10, dt=0.01, nsteps=2000, nwrite=2, tol=1e-15)
"""DSLAB: the derivation job's slab (predictions.json P03): 2 mm + 2 mm, 10 + 10 cells, dt 0.01 s, 2000 steps."""


def x_bc_smoke():
    """First runs of the derived boundary condition and of the guard (12)."""
    pts = [('xbc_p1_t30_n16', 'patch', dict(P1, theta=30, nxa=16, nxi=16, ny=16, coupling='conormalbc', cn_lib=CN)),
           ('xbc_p1s_t30', 'patch', dict(P1, theta=30, nxa=16, nxi=16, ny=4, ybc='gradient', coupling='conormalbc',
                                         cn_lib=CN)),
           ('xbc_p1_t0_n16', 'patch', dict(P1, theta=0, nxa=16, nxi=16, ny=16, coupling='conormalbc', cn_lib=CN)),
           # the derived BC with its tangential term switched off must equal the stock directional option
           # guard: derived BC with the scalar option -> FatalError at the first evaluation
           ('xguard_bc_scalar', 'patch', dict(P1, theta=45, nxa=8, nxi=8, ny=8, coupling='scalarbc', cn_lib=CN)),
           # guard function object with the STOCK condition
           ('xguard_fo_strict', 'patch', dict(P1, theta=45, nxa=8, nxi=8, ny=8, coupling='scalar', guard='strict',
                                              cn_lib=CN)),
           ('xguard_fo_error', 'patch', dict(P1, theta=45, nxa=8, nxi=8, ny=8, coupling='scalar', guard='error',
                                             cn_lib=CN)),
           ('xguard_fo_warn', 'patch', dict(P1, theta=45, nxa=8, nxi=8, ny=8, coupling='scalar', guard='warn',
                                            cn_lib=CN)),
           ('xguard_fo_directional', 'patch', dict(P1, theta=45, nxa=8, nxi=8, ny=8, coupling='directional',
                                                   guard='strict', cn_lib=CN)),
           # (k k k) through constAnIso is "anisotropic" for the code: the guard must fire
           ('xguard_fo_kkk', 'slab', dict(k1=2, k2=2, k3=2, theta=0, coupling='scalar', guard='strict', cn_lib=CN,
                                          nxa=8, nxi=8, nsteps=20)),
           # audit function objects on the smoke patch test (acceptance-test table)
           ('xaud_p1_t30_scalar', 'patch', dict(P1, theta=30, nxa=16, nxi=16, ny=16, coupling='scalar', audit=1, cn_lib=CN)),
           ('xaud_p1_t30_directional', 'patch', dict(P1, theta=30, nxa=16, nxi=16, ny=16, coupling='directional', audit=1,
                                                     cn_lib=CN)),
           ('xaud_p1_t30_conormalbc', 'patch', dict(P1, theta=30, nxa=16, nxi=16, ny=16, coupling='conormalbc', audit=1,
                                                    cn_lib=CN))]
    return pts


def x_pred():
    """Replicas of the derivation job's configurations, generated with the design job's
    generator, so that predictions.json is tested by runs of this job (about 60)."""
    pts = []
    for th in (0, 30, 45, 60, 90):                                      # P03, P06, P08
        for cpl in ('scalar', 'directional'):
            pts.append((f'xd_slab_{cpl}_t{th}', 'slab', dict(DSLAB, theta=th, coupling=cpl)))
    for cpl in ('scalar', 'directional'):                               # P04
        pts.append((f'xd_slab_{cpl}_kkk', 'slab', dict(DSLAB, k1=2, k2=2, k3=2, theta=30, coupling=cpl)))
    for n in (5, 40):                                                   # P05 mesh
        pts.append((f'xd_slab_scalar_t90_n{n}', 'slab', dict(DSLAB, theta=90, coupling='scalar', nxa=n, nxi=n)))
    for dt, ns in ((0.04, 500), (0.0025, 8000)):                        # P05 time step
        pts.append((f'xd_slab_scalar_t90_dt{dt}', 'slab', dict(DSLAB, theta=90, coupling='scalar', dt=dt, nsteps=ns)))
    for kf in (0.15, 5.0):                                              # P05 neighbour
        pts.append((f'xd_slab_scalar_t90_kf{kf}', 'slab', dict(DSLAB, theta=90, coupling='scalar', ki=kf)))
    for RL in (1e-4, 1e-3):                                             # P05 layer resistance (emulator-only so far)
        pts.append((f'xd_slab_scalar_t90_RL{RL}', 'slab', dict(DSLAB, theta=90, coupling='scalar', Rlayer=RL)))
        pts.append((f'xd_slab_directional_t90_RL{RL}', 'slab', dict(DSLAB, theta=90, coupling='directional', Rlayer=RL)))
    for no in (2, 3):                                                   # P10
        pts.append((f'xd_slab_directional_t90_o{no}', 'slab', dict(DSLAB, theta=90, coupling='directional', nouter=no)))
        pts.append((f'xd_slab_scalar_t90_o{no}', 'slab', dict(DSLAB, theta=90, coupling='scalar', nouter=no)))
    pts.append(('xd_slab_directional_t90_rev', 'slab', dict(DSLAB, theta=90, coupling='directional', order='aniso iso')))
    for cpl in ('scalar', 'directional'):                               # P09 closed pair, 1 mm + 1 mm, 5 + 5 cells
        pts.append((f'xd_eq_{cpl}', 'slab', dict(DSLAB, theta=90, coupling=cpl, La=0.001, Li=0.001, nxa=5, nxi=5,
                                                 dt=0.01, nsteps=12000, nwrite=1)))
    for cpl in ('scalar', 'directional'):                               # P07 steady slab, fixed ends
        for th in (45, 90):
            for n in (10, 40):
                pts.append((f'xd_slabfix_{cpl}_t{th}_n{n}', 'slab',
                            dict(DSLAB, theta=th, coupling=cpl, nxa=n, nxi=n, xbc='DD', ic='uniform', Ta0=320, TL=350,
                                 TR=300, rhoA=1, cpA=1000, rhoI=1, cpI=1000, dt=0.05, nsteps=3000, nwrite=2)))
    for n in (5, 10, 20, 40):                                           # P18 Cp(T), four passes, audit objects
        pts.append((f'xd_slab_cpT_n{n}', 'slab',
                    dict(DSLAB, theta=90, coupling='directional', nxa=n, nxi=n, cpI=824.66, cpI1=3.6, dt=0.0025,
                         nsteps=400, nouter=4, audit=1, cn_lib=CN)))
    pts.append(('xd_onestep_cpT', 'slab', dict(DSLAB, theta=90, coupling='directional', nxa=5, nxi=5, cpI=824.66,   # P02
                                               cpI1=3.6, dt=0.0025, nsteps=1, nwrite=1)))
    for cpl in ('scalar', 'directional'):                               # P21, P22 per-step function-object books
        for th in (45, 90):
            pts.append((f'xd_aud_{cpl}_t{th}', 'slab', dict(DSLAB, theta=th, coupling=cpl, audit=1, cn_lib=CN)))
    pts.append(('xd_aud_scalar_kkk', 'slab', dict(DSLAB, k1=2, k2=2, k3=2, theta=30, coupling='scalar', audit=1, cn_lib=CN)))
    pts.append(('xd_aud_scalar_t90_o4', 'slab', dict(DSLAB, theta=90, coupling='scalar', audit=1, cn_lib=CN, nouter=4)))
    pts.append(('xd_aud_directional_t90_o4', 'slab', dict(DSLAB, theta=90, coupling='directional', audit=1, cn_lib=CN,
                                                          nouter=4)))
    # P14 sheared closed form at 45 degrees with B = 1000 K/m; P15 coincidence A = B at 45 degrees
    pts.append(('xd_p14_shear_t45_B1000', 'patch', dict(P1, theta=45, B=1000, nxa=16, nxi=16, ny=4, ybc='gradient',
                                                        coupling='directional')))
    for cpl in ('scalar', 'directional'):
        pts.append((f'xd_p15_t45_AeqB_{cpl}', 'patch', dict(P1, theta=45, A=1000, B=1000, nxa=16, nxi=16, ny=16,
                                                            coupling=cpl)))
    # P25: two non-orthogonal correctors requested for the solid; count the h solves
    pts.append(('xd_p25_nnonorth2', 'patch', dict(P1, theta=30, nxa=16, nxi=16, ny=16, coupling='directional', nnonorth=2)))
    # P17: closed strip, zero-gradient aniso end, tangential gradient B: the end wall is not adiabatic
    P17 = dict(k1=28, k2=0.8, k3=0.8, ki=1.0, La=0.004, Li=0.006, H=0.002, ny=4, nxa=16, nxi=24, rhoA=2000, cpA=1000,
               rhoI=1000, cpI=1500, xbc='NN', ic='uniform', Ta0=320, B=1000, dt=0.05, nsteps=400, nwrite=4, allow_leak=1,
               audit=1, cn_lib=CN)
    for th in (0, 45):
        for cpl in ('directional', 'conormalbc'):
            pts.append((f'xd_p17_box_t{th}_{cpl}', 'slab', dict(P17, theta=th, coupling=cpl)))
    return pts


def x_p1bc():
    """The corrected boundary condition over the P1 grid: sheared strip at every ratio and
    angle, all-Dirichlet patch test on 16 cells at every ratio and angle, 8/32/64 cells at
    selected points; coupling passes (118)."""
    pts = []
    for r in RATIOS:
        for th in ANGLES:
            base = dict(k1=K_LOW * r, k2=K_LOW, k3=K_LOW, theta=th, ki=1.0, A=1000, B=500, coupling='conormalbc', cn_lib=CN)
            pts.append((f'p1s_r{r}_t{th}_conormalbc', 'patch', dict(base, ybc='gradient', nxa=16, nxi=16, ny=4, nsteps=600)))
            pts.append((f'p1_r{r}_t{th}_conormalbc_n16', 'patch', dict(base, nxa=16, nxi=16, ny=16, nsteps=400)))
    for n in (8, 32):
        for th in ANGLES:
            pts.append((f'p1_r35_t{th}_conormalbc_n{n}', 'patch',
                        dict(P1, theta=th, coupling='conormalbc', cn_lib=CN, nxa=n, nxi=n, ny=n, nsteps=25 * n)))
        pts.append((f'p1_r1000_t45_conormalbc_n{n}', 'patch',
                    dict(P1, k1=800.0, theta=45, coupling='conormalbc', cn_lib=CN, nxa=n, nxi=n, ny=n, nsteps=100 * n)))
    for th in (0, 30, 60):
        pts.append((f'p1_r35_t{th}_conormalbc_o3', 'patch', dict(P1, theta=th, coupling='conormalbc', cn_lib=CN, nouter=3,
                                                                 nxa=16, nxi=16, ny=16)))
        pts.append((f'p1_r35_t{th}_conormalbc_phi20', 'patch', dict(P1, theta=th, coupling='conormalbc', cn_lib=CN, phi=20,
                                                                    nxa=16, nxi=16, ny=16)))
    pts.append(('p1_r35_t45_conormalbc_n64', 'patch', dict(P1, theta=45, coupling='conormalbc', cn_lib=CN, nxa=64, nxi=64,
                                                           ny=64, nsteps=1600)))
    return pts


RING = dict(kx=28.0, ky=0.8, kz=0.8, ko=1.0, T0=350.0, Ex=100.0, Ey=500.0)


def _ring(name, lev, **kw):
    a = dict(RING, nra=10 * lev, nri=12 * lev, nth=8 * lev, nsteps=600 * lev)
    a.update(kw)
    if a['coupling'] == 'conormal':
        a['conormal_lib'] = FO
    if a['coupling'] in ('conormalbc', 'scalarbc'):
        a['cn_lib'] = CN
    return (name, 'ring', a)


def x_ring():
    """P5, curved interface with Cartesian orthotropy (ratio 35): non-linear exact solution
    (aq = 5e4) on levels 1, 2, 3, 4 for the stock directional option and the corrected BC;
    scalar option and function-object variant on levels 1, 2; linear interior (aq = 0),
    rotated axes, other ratios, second-harmonic data (about 40)."""
    pts = []
    for lev in (1, 2, 3, 4):
        for cpl in ('directional', 'conormalbc'):
            pts.append(_ring(f'ring_r35_q_{cpl}_L{lev}', lev, coupling=cpl, aq=5.0e4))
    for lev in (1, 2):
        for cpl in ('scalar', 'conormal'):
            pts.append(_ring(f'ring_r35_q_{cpl}_L{lev}', lev, coupling=cpl, aq=5.0e4))
    for lev in (1, 2):
        for cpl in ('directional', 'conormalbc'):
            pts.append(_ring(f'ring_r35_lin_{cpl}_L{lev}', lev, coupling=cpl, aq=0.0))
    # the scalar option diverged on the ring with the pseudo-time step 0.05 s (negative temperature after
    # 20 steps); repeated with a step ten and a hundred times smaller to see whether that is a step-size limit
    pts.append(_ring('ring_r35_q_scalar_L1_dt0.005', 1, coupling='scalar', aq=5.0e4, dt=0.005, nsteps=6000))
    pts.append(_ring('ring_r35_q_scalar_L1_dt0.0005', 1, coupling='scalar', aq=5.0e4, dt=0.0005, nsteps=20000))
    for lev in (1, 2):
        pts.append(_ring(f'ring_r35_q_psi25_conormalbc_L{lev}', lev, coupling='conormalbc', aq=5.0e4, psi=25.0))
        pts.append(_ring(f'ring_r35_q_psi25_directional_L{lev}', lev, coupling='directional', aq=5.0e4, psi=25.0))
    for r in (1, 3, 10, 100):
        for cpl in ('directional', 'conormalbc'):
            pts.append(_ring(f'ring_r{r}_q_{cpl}_L2', 2, coupling=cpl, aq=5.0e4, kx=K_LOW * r))
    # ratios 200 and 1000 need a smaller gradient along the high-conductivity axis: with E_x = 100 K/m the
    # exact outer field of ratio 1000 goes below 0 K (the first attempt stopped in the thermo package after
    # two steps for that reason and was removed from the archive)
    for r, Ex in ((200, 20.0), (1000, 5.0)):
        for cpl in ('directional', 'conormalbc'):
            pts.append(_ring(f'ring_r{r}_q_{cpl}_L2', 2, coupling=cpl, aq=5.0e4, kx=K_LOW * r, Ex=Ex))
    for cpl in ('directional', 'conormalbc'):                 # one finer level for the order
        pts.append(_ring(f'ring_r35_q_{cpl}_L6', 6, coupling=cpl, aq=5.0e4))
    for lev in (1, 2):
        pts.append(_ring(f'ring_r35_harm2_directional_L{lev}', lev, coupling='directional', bcdata='harm2'))
    pts.append(_ring('ring_r35_harm2_conormalbc_L2', 2, coupling='conormalbc', bcdata='harm2'))
    return pts


def x_annulus_bc():
    """P2 with the corrected BC (the normal is a principal direction: must equal the stock
    directional option), ratio 35, both variants, levels 1, 2, 4 (6)."""
    pts = []
    for var, kk, sec, nth in (('A', dict(kr=0.8, kth=0.8, kz=28.0), 90, 16), ('B', dict(kr=0.8, kth=28.0, kz=28.0), 360, 8)):
        for lev in (1, 2, 4):
            pts.append((f'p2{var}_r35_conormalbc_L{lev}', 'annulus',
                        dict(kk, sector=sec, coupling='conormalbc', cn_lib=CN, nra=10 * lev, nri=12 * lev, nth=nth * lev,
                             nsteps=600 * lev)))
    return pts


def x_p3bc():
    """Transient with a tangential gradient (design block P3-sheared) with the corrected BC,
    and with three coupling passes for the function-object variant and the BC (8)."""
    P3 = dict(k1=28, k2=0.8, k3=0.8, theta=30, ki=0.15, La=0.004, Li=0.006, H=0.002, rhoA=2000, cpA=1000, rhoI=1000,
              cpI=1500, xbc='DD', ic='uniform', Ta0=300, TL=350, TR=300, B=2000, nxa=32, nxi=48, ny=4, nwrite=4)
    pts = []
    for m in (1, 2, 4):
        pts.append((f'p3s_conormalbc_dt{m}', 'slab', dict(P3, coupling='conormalbc', cn_lib=CN, dt=0.4 / m, nsteps=100 * m)))
    for cpl, extra in (('directional', {}), ('conormal', dict(conormal_lib=FO)), ('conormalbc', dict(cn_lib=CN))):
        pts.append((f'p3s_{cpl}_dt2_o3', 'slab', dict(P3, coupling=cpl, dt=0.2, nsteps=200, nouter=3, **extra)))
    for cpl, extra in (('conormal', dict(conormal_lib=FO)), ('conormalbc', dict(cn_lib=CN))):
        pts.append((f'p3s_{cpl}_dt8', 'slab', dict(P3, coupling=cpl, dt=0.05, nsteps=800, **extra)))
    return pts


def x_audit_steady():
    """Acceptance-test table: stock and corrected flux objects on the annulus (4)."""
    pts = []
    for cpl in ('scalar', 'directional'):
        pts.append((f'xaud_p2A_r35_{cpl}_L1', 'annulus', dict(kr=0.8, kth=0.8, kz=28.0, sector=90, coupling=cpl, nra=10,
                                                              nri=12, nth=16, audit=1, cn_lib=CN)))
    for cpl in ('scalar', 'directional', 'conormalbc'):
        pts.append(_ring(f'xaud_ring_r35_q_{cpl}_L2', 2, coupling=cpl, aq=5.0e4, audit=1, cn_lib=CN))
    return pts


def x_equiv():
    """Three points already run through the design job's own Allrun, repeated through this
    driver in a separate folder to show the two paths give identical numbers."""
    ids = ('p1s_r35_t30_directional', 'p3_directional_o1_L1_dt1', 'p1s_r35_t30_conormal')
    return _design(lambda p: p['id'] in ids)


def x_guard2():
    """Guard function object after the fix of its non-strict mode (see RESULTS.md, deviations)."""
    return [p for p in x_bc_smoke() if p[0].startswith('xguard_fo')]


def x_meshindep():
    """Flux-ratio figure: scalar option, normal along the 0.8 axis (predicted 35.0286), the
    derivation job's slab on 5 to 160 cells per region with one and with four passes, and
    five time steps on 10 cells (17)."""
    pts = []
    for n in (5, 10, 20, 40, 80, 160):
        for no in (1, 4):
            pts.append((f'xm_slab_scalar_t90_n{n}_o{no}', 'slab', dict(DSLAB, theta=90, coupling='scalar', nxa=n, nxi=n,
                                                                        nouter=no)))
    for dt, ns in ((0.04, 500), (0.02, 1000), (0.005, 4000), (0.0025, 8000), (0.00125, 16000)):
        pts.append((f'xm_slab_scalar_t90_n10_dt{dt}', 'slab', dict(DSLAB, theta=90, coupling='scalar', dt=dt, nsteps=ns)))
    return pts


def x_early():
    """One instance of each added case type, run first to catch generator errors early
    (the ids are those of the full blocks, so the results are kept)."""
    want = ('ring_r35_q_directional_L1', 'ring_r35_q_conormalbc_L1', 'ring_r35_q_conormal_L1', 'ring_r35_q_scalar_L1',
            'ring_r35_harm2_directional_L1', 'xd_slab_cpT_n5', 'xd_onestep_cpT', 'xd_slab_scalar_t90_RL0.0001',
            'xd_p17_box_t45_conormalbc', 'xd_aud_scalar_t90', 'xd_slabfix_scalar_t90_n10', 'xd_p25_nnonorth2',
            'p2A_r35_conormalbc_L1', 'xaud_p2A_r35_directional_L1', 'p3s_conormalbc_dt2', 'p3s_conormalbc_dt2_o3',
            'ring_r35_q_directional_L2', 'ring_r35_q_conormalbc_L2')
    allp = {p[0]: p for f in (x_ring, x_pred, x_annulus_bc, x_audit_steady, x_p3bc) for p in f()}
    return [allp[w] for w in want]


def x_more():
    """Aligned patch tests (0 and 90 degrees) with the audit objects (acceptance-test table) (2)."""
    return [('xaud_p1_t0_directional', 'patch', dict(P1, theta=0, nxa=16, nxi=16, ny=16, coupling='directional', audit=1,
                                                     cn_lib=CN)),
            ('xaud_p1_t90_directional', 'patch', dict(P1, theta=90, nxa=16, nxi=16, ny=16, coupling='directional', audit=1,
                                                      cn_lib=CN))]


def x_stab():
    """Stability of the pseudo-time march at high anisotropy ratio: the corrected condition
    diverged in the all-Dirichlet patch test at ratio 100 (15 and 30 degrees) with the design's
    step of 0.05 s.  Same points with smaller steps, with the function-object variant, with the
    stock options, and with a larger heat capacity (smaller diffusion number at the same step)."""
    pts = []
    base = dict(k1=80.0, k2=0.8, k3=0.8, ki=1.0, A=1000, B=500, nxa=16, nxi=16, ny=16)
    for th in (15, 30):
        for cpl, extra in (('directional', {}), ('conormal', dict(conormal_lib=FO)), ('conormalbc', dict(cn_lib=CN))):
            for dt, ns in ((0.05, 400), (0.005, 2000), (0.0005, 8000)):
                if cpl == 'directional' and dt != 0.05:
                    continue
                pts.append((f'xs_p1_r100_t{th}_{cpl}_dt{dt}', 'patch', dict(base, theta=th, coupling=cpl, dt=dt, nsteps=ns,
                                                                             _maxcont=(4 if dt == 0.05 else 1), **extra)))
    # neighbour ten and a hundred times more conductive: does the limit depend on k_i ?
    for ki in (10.0, 100.0):
        pts.append((f'xs_p1_r100_t30_conormalbc_ki{ki:g}', 'patch', dict(base, theta=30, coupling='conormalbc', cn_lib=CN, ki=ki)))
    # physical heat capacities (rho c_p = 2.0e6 and 1.5e6 J/m3/K instead of 1e3), started from the exact
    # field: is the divergence a property of the pseudo-time march only?  100 s and 400 s of physical time.
    phys = dict(rhoA=2000.0, cpA=1000.0, rhoI=1000.0, cpI=1500.0, init='exact')
    for r, th in ((100, 30), (1000, 30), (1000, 45)):
        b2 = dict(base, k1=0.8 * r, theta=th, coupling='conormalbc', cn_lib=CN, **phys)
        pts.append((f'xs_p1_r{r}_t{th}_conormalbc_phys_dt0.05', 'patch', dict(b2, dt=0.05, nsteps=2000, _maxcont=0)))
        pts.append((f'xs_p1_r{r}_t{th}_conormalbc_phys_dt0.4', 'patch', dict(b2, dt=0.4, nsteps=1000, _maxcont=0)))
    # the same pseudo-time settings as the design, started from the exact field (round-off perturbation only)
    pts.append(('xs_p1_r1000_t30_conormalbc_exactinit', 'patch', dict(base, k1=800.0, theta=30, coupling='conormalbc',
                                                                     cn_lib=CN, init='exact', nsteps=1200)))
    return pts


def x_annulus_L8():
    """Finest annulus level (80+96 radial cells) for the scalar option and the corrected BC, variant A (2)."""
    return [(f'p2A_r35_{cpl}_L8', 'annulus', dict(kr=0.8, kth=0.8, kz=28.0, sector=90, coupling=cpl, nra=80, nri=96, nth=128,
                                                 nsteps=4800, **({'cn_lib': CN} if cpl == 'conormalbc' else {})))
            for cpl in ('scalar', 'conormalbc')]


def x_ringstab():
    """Ring at ratio 200: both the stock directional option and the corrected one diverged at the design's
    pseudo-time step.  Same case with steps ten and a hundred times smaller (4)."""
    pts = []
    for cpl in ('directional', 'conormalbc'):
        for dt, ns in ((0.005, 6000), (0.0005, 20000)):
            p = _ring(f'ring_r200_q_{cpl}_L2_dt{dt}', 2, coupling=cpl, aq=5.0e4, kx=K_LOW * 200, Ex=20.0, dt=dt, nsteps=ns)
            p[2]['_maxcont'] = 1
            pts.append(p)
    return pts
