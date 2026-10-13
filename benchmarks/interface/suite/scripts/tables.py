#!/usr/bin/env python3
"""
tables.py -- the tables of RESULTS.md, built from results.csv, the result.json files and
predictions_tested.json.  Imported by make_results.py; `python3 tables.py` prints them all.
No number in these tables is typed by hand.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact as ex          # noqa: E402
import blocks               # noqa: E402

BENCH = os.path.dirname(HERE)
RATIOS = (1, 3, 10, 35, 100, 200, 1000)
ANGLES = (0, 15, 30, 45, 60, 75, 90)


def fl(v):
    return float(v) if v not in ('', None) else None


class DB:
    def __init__(self, cases=None):
        self.cases = cases or os.path.join(BENCH, 'cases')
        self.rows = list(csv.DictReader(open(os.path.join(BENCH, 'results.csv'))))
        self.by = {r['id']: r for r in self.rows}
        self._j = {}

    def j(self, cid):
        if cid not in self._j:
            f = os.path.join(self.cases, cid, 'result.json')
            self._j[cid] = json.load(open(f)) if os.path.exists(f) else None
        return self._j[cid]

    def v(self, cid, col):
        r = self.by.get(cid)
        return fl(r[col]) if r else None

    def sel(self, **kw):
        out = []
        for r in self.rows:
            if all((r[k] == v if not callable(v) else v(r[k])) for k, v in kw.items()):
                out.append(r)
        return out


def md(header, rows, align=None):
    align = align or ['---'] * len(header)
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def e(v, d=2):
    if v is None:
        return 'n/a'
    return f'{v:.{d}e}'


def g(v, d=6):
    if v is None:
        return 'n/a'
    return f'{v:.{d}g}'


def order(h, err):
    return [math.log(err[i] / err[i + 1]) / math.log(h[i] / h[i + 1]) for i in range(len(err) - 1)
            if err[i] and err[i + 1]]


# ------------------------------------------------------------------ run inventory


def inventory(db):
    design = {}
    for p in blocks.M.points():
        design.setdefault(p['tag'], [0, 0])
        design[p['tag']][0] += 1
        if p['id'] in db.by:
            design[p['tag']][1] += 1
    rows = [(t, n, k, n - k) for t, (n, k) in design.items()]
    rows.append(('**total design matrix**', sum(r[1] for r in rows), sum(r[2] for r in rows), sum(r[3] for r in rows)))
    t1 = md(['design block', 'points in the design', 'run here', 'left out'], rows)
    added = {}
    for r in db.rows:
        if r['family'] == 'added':
            a = added.setdefault(r['block'] or '(other)', [0, 0])
            a[0] += 1
            a[1] += r['status'] == 'aborted'
    doc = {name: (getattr(blocks, name).__doc__ or '').strip().split('\n')[0] for name in dir(blocks) if name.startswith('x_')}
    t2 = md(['added block', 'runs', 'of which stopped (guard tests and one divergence)', 'purpose'],
            [(b, n, k, doc.get(b, '')) for b, (n, k) in sorted(added.items())])
    cpu = sum(fl(r['cpu_s']) or 0 for r in db.rows)
    return t1, t2, dict(total=len(db.rows), design=sum(1 for r in db.rows if r['family'] == 'design'),
                        added=sum(1 for r in db.rows if r['family'] == 'added'),
                        aborted=sum(1 for r in db.rows if r['status'] == 'aborted'), cpu_s=cpu)


# ------------------------------------------------------------------ H1 flux ratio


def h1_ratio(db):
    rows, worst = [], 0.0
    for r in RATIOS:
        line = [r]
        for th in (0, 30, 45, 60, 90):
            m, p = db.v(f'p4_r{r}_t{th}_scalar', 'energy_ratio_destaggered'), db.v(f'p4_r{r}_t{th}_scalar', 'ratio_predicted')
            if m is None:
                line.append('n/a')
                continue
            worst = max(worst, abs(m / p - 1))
            line.append(f'{m:.9g}')
        rows.append(line)
    t = md(['k1/k2', 'theta = 0', '30', '45', '60', '90'], rows)
    # predicted values for the same grid
    prow = []
    for r in RATIOS:
        k = (0.8 * r, 0.8, 0.8)
        prow.append([r] + [f'{ex.frob(k) / ex.knn_knt(k, th)[0]:.9g}' for th in (0, 30, 45, 60, 90)])
    tp = md(['k1/k2', 'theta = 0', '30', '45', '60', '90'], prow)
    wd = 0.0
    for r in RATIOS:
        for th in (0, 30, 45, 60, 90):
            m = db.v(f'p4_r{r}_t{th}_directional', 'energy_ratio_destaggered')
            if m is not None:
                wd = max(wd, abs(m - 1))
    return t, tp, worst, wd


def h1_other(db):
    rows = []
    for kc in (0.1, 0.4, 1.6, 5.0, 80.0):
        c = f'p4_kc{kc}'
        rows.append((f'function, k_c = {kc:g}', '(28, 0.8, 0.8), normal along 0.8', g(db.v(c, 'ratio_predicted'), 10),
                     g(db.v(c, 'energy_ratio_destaggered'), 10), g(db.v(c, 'energy_ratio_raw'), 8)))
    for k in (0.5, 2.0, 50.0):
        for cpl in ('scalar', 'directional'):
            c = f'p4_kkk{k}_{cpl}'
            rows.append((cpl, f'({k:g}, {k:g}, {k:g})', g(db.v(c, 'ratio_predicted'), 10),
                         g(db.v(c, 'energy_ratio_destaggered'), 10), g(db.v(c, 'energy_ratio_raw'), 8)))
    t1 = md(['coupling', 'principal conductivities', 'predicted k_c/(n.K.n)', 'measured, staggering term removed',
             'measured, raw one pass'], rows)
    rows = []
    for ki in (0.15, 5.0):
        for lev, m in ((1, 1), (2, 1), (1, 4), (4, 4)):
            c = f'p4_ki{ki}_L{lev}_dt{m}'
            rows.append((ki, f'{16 * lev}+{24 * lev}', 0.05 / m, g(db.v(c, 'energy_ratio_raw'), 9),
                         g(db.v(c, 'energy_ratio_destaggered'), 12), g(db.v(c, 'err_solver_minus_emulator'), 2)))
    t2 = md(['neighbour k', 'cells', 'dt [s]', 'raw ratio, one pass', 'staggering term removed', 'solver minus emulator [K]'], rows)
    return t1, t2


def meshindep(db):
    pred = ex.frob((28.0, 0.8, 0.8)) / 0.8
    rows = []
    for n in (5, 10, 20, 40, 80, 160):
        a, b = f'xm_slab_scalar_t90_n{n}_o1', f'xm_slab_scalar_t90_n{n}_o4'
        rows.append((f'{n}+{n}', 0.01, g(db.v(a, 'energy_ratio_raw'), 10), g(db.v(a, 'energy_ratio_destaggered'), 13),
                     g(db.v(b, 'energy_ratio_raw'), 13)))
    for dt in (0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125):
        a = 'xm_slab_scalar_t90_n10_o1' if dt == 0.01 else f'xm_slab_scalar_t90_n10_dt{dt}'
        rows.append(('10+10', dt, g(db.v(a, 'energy_ratio_raw'), 10), g(db.v(a, 'energy_ratio_destaggered'), 13), ''))
    t = md(['cells', 'dt [s]', 'raw ratio, one pass', 'one pass, staggering term removed', 'raw ratio, four passes'], rows)
    st = []
    for lev in (1, 2, 4, 8):
        c = f'p2A_r35_scalar_L{lev}'
        j = db.j(c)
        if j:
            m = j['measured_Qp']
            st.append((f'annulus A, {10 * lev}+{12 * lev} radial cells', g(m['interface_into_iso'] / m['interface_out_of_aniso'], 13)))
    for n in (8, 16, 32):
        c = f'p1_r35_t90_scalar_n{n}'
        if db.v(c, 'ratio_steady') is not None:
            st.append((f'patch test at 90 degrees, {n} cells per side', g(db.v(c, 'ratio_steady'), 13)))
    c = 'p1s_r35_t90_scalar'
    if db.v(c, 'ratio_steady') is not None:
        st.append(('sheared strip at 90 degrees, 16 cells', g(db.v(c, 'ratio_steady'), 13)))
    t2 = md(['steady problem', 'heat rate into iso / normal heat rate out of aniso'], st)
    return t, t2, pred


# ------------------------------------------------------------------ P1


def p1_grid(db, cpl, kind='patch', n=16, col='Linf_exact'):
    rows = []
    for r in RATIOS:
        line = [r]
        for th in ANGLES:
            c = f'p1_r{r}_t{th}_{cpl}_n{n}' if kind == 'patch' else f'p1s_r{r}_t{th}_{cpl}'
            v = db.v(c, col)
            if c in db.by and db.by[c]['status'] == 'aborted':
                line.append('diverged')
            else:
                line.append('not run' if v is None else (f'{v:.3f}' if abs(v) >= 1e-3 else f'{v:.0e}'))
        rows.append(line)
    return md(['k1/k2'] + [f'{t} deg' for t in ANGLES], rows)


def p1_stats(db):
    """Largest deviations over the whole P1 grid."""
    out = {}
    for kind in ('patch', 'sheared'):
        for cpl in ('scalar', 'directional', 'conormal', 'conormalbc'):
            errs, imb, sch, steady, n, obl, div = [], [], [], [], 0, [], []
            for r in RATIOS:
                for th in ANGLES:
                    c = f'p1_r{r}_t{th}_{cpl}_n16' if kind == 'patch' else f'p1s_r{r}_t{th}_{cpl}'
                    row = db.by.get(c)
                    if row and row['status'] == 'aborted':
                        div.append((r, th))
                    if not row or row['status'] != 'ok':
                        continue
                    n += 1
                    errs.append(fl(row['Linf_exact']))
                    if fl(row['imbalance_predicted']) is not None:
                        imb.append(abs(fl(row['imbalance']) - fl(row['imbalance_predicted'])))
                    if fl(row['Linf_scheme']) is not None:
                        sch.append(fl(row['Linf_scheme']))
                    steady.append(fl(row['steady_change']) or 0.0)
                    if th not in (0, 90) and r > 1:
                        obl.append(fl(row['Linf_exact']))
            out[(kind, cpl)] = dict(n=n, diverged=div, max_err=max(errs) if errs else None, min_oblique_err=min(obl) if obl else None,
                                    max_oblique_err=max(obl) if obl else None,
                                    max_imbalance_dev=max(imb) if imb else None, max_scheme_err=max(sch) if sch else None,
                                    max_steady=max(steady) if steady else None)
    return out


def p1_mesh(db):
    rows = []
    for r, th in ((35, 30), (35, 45), (35, 60), (1000, 45)):
        for cpl in ('directional', 'scalar', 'conormalbc'):
            line = [r, th, cpl]
            for n in (8, 16, 32, 64):
                cid = f'p1_r{r}_t{th}_{cpl}_n{n}'
                v = db.v(cid, 'Linf_exact')
                if cid in db.by and db.by[cid]['status'] == 'aborted':
                    line.append('diverged')
                else:
                    line.append('' if v is None else (f'{v:.4f}' if v >= 1e-3 else f'{v:.1e}'))
            v = db.v(f'p1_r{r}_t{th}_{cpl}_n16', 'imbalance')
            line.append('' if v is None else (f'{v:.4f}' if abs(v) >= 1e-3 else f'{v:.1e}'))
            pv = db.v(f'p1_r{r}_t{th}_{cpl}_n16', 'imbalance_predicted')
            line.append('' if pv is None or cpl == 'scalar' else f'{pv:.4f}')
            rows.append(line)
    return md(['k1/k2', 'theta', 'coupling', 'Linf [K], 8 cells', '16', '32', '64', 'created at interface [W/m], 16 cells',
               'predicted K_nt B H'], rows)


def frame(db):
    rows = []
    for th in (0, 30, 60):
        base = f'p1_r35_t{th}_directional_n16'
        for cid, what in ((base, 'reference'), (f'p1_r35_t{th}_phi20', 'frame rotated 20 deg'),
                          (f'p1_r35_t{th}_phi45', 'frame rotated 45 deg'), (f'p1_r35_t{th}_o3', 'three passes'),
                          (f'p1_r35_t{th}_conormalbc_phi20', 'corrected, frame rotated 20 deg'),
                          (f'p1_r35_t{th}_conormalbc_o3', 'corrected, three passes')):
            if cid in db.by:
                rows.append((th, what, g(db.v(cid, 'Linf_exact'), 10), g(db.v(cid, 'imbalance'), 10)))
    return md(['theta', 'variant', 'Linf error [K]', 'created at interface [W/m]'], rows)


# ------------------------------------------------------------------ annulus and ring


def annulus(db):
    rows, orders = [], {}
    for var in ('A', 'B'):
        for cpl in ('directional', 'conormalbc', 'scalar'):
            H, E2, Ei, Es = [], [], [], []
            for lev in (1, 2, 4, 8):
                c = f'p2{var}_r35_{cpl}_L{lev}'
                j = db.j(c)
                if not j:
                    continue
                m = j['measured_Qp']
                es = j['error'].get('scheme', {}).get('all', {})
                rows.append((var, cpl, f"{j['cells']['aniso']}+{j['cells']['iso']}", e(j['error']['exact']['all']['L2'], 3),
                             e(j['error']['exact']['all']['Linf'], 3), e(es.get('Linf'), 3) if cpl == 'scalar' else '',
                             g(m['interface_out_of_aniso'], 8), g(m['interface_into_iso'], 8),
                             g(m['interface_into_iso'] / m['interface_out_of_aniso'], 11),
                             e(j['interface']['imbalance'], 1)))
                H.append(1.0 / lev); E2.append(j['error']['exact']['all']['L2']); Ei.append(j['error']['exact']['all']['Linf'])
                Es.append(es.get('Linf'))
            if len(H) > 1:
                orders[(var, cpl)] = dict(L2=order(H, E2), Linf=order(H, Ei),
                                          scheme_Linf=order(H, Es) if all(Es) else None, levels=[1 / h for h in H])
    t = md(['variant', 'coupling', 'cells', 'L2 vs exact [K]', 'Linf vs exact [K]', 'Linf vs own problem [K]',
            "Q' out of aniso", "Q' into iso", 'ratio', 'imbalance [W/m]'], rows)
    return t, orders


def annulus_ratios(db):
    rows = []
    for var in ('A', 'B'):
        for r in RATIOS:
            for cpl in ('scalar', 'directional'):
                c = f'p2{var}_r{r}_{cpl}_L1'
                j = db.j(c)
                if not j:
                    continue
                m = j['measured_Qp']
                rows.append((var, r, cpl, g(db.v(c, 'ratio_predicted'), 10),
                             g(m['interface_into_iso'] / m['interface_out_of_aniso'], 12),
                             e(j['error']['exact']['all']['Linf'], 3)))
    return md(['variant', 'k_high/k_low', 'coupling', 'predicted k_c/k_r', "measured Q' into iso / Q' out of aniso",
               'Linf vs exact [K]'], rows)


def trap(db):
    rows = []
    for nth in (16, 32, 64, 128):
        j = db.j(f'p2C_r35_nth{nth}')
        if j:
            m = j['measured_Qp']
            rows.append((nth, g(m['inner_wall'], 6), g(m['interface_out_of_aniso'], 6), g(j['exact']['Qp'], 6),
                         e(j['error']['exact']['all']['Linf'], 3), e(j['interface']['imbalance'], 1)))
    return md(['cells over 90 deg', "Q' at inner wall", "Q' across interface", "exact Q'", 'Linf [K]',
               'interface imbalance [W/m]'], rows)


def ring(db):
    rows, orders = [], {}
    for name, label in (('ring_r35_q', 'quadratic interior, axes aligned with the mesh'),
                        ('ring_r35_q_psi25', 'quadratic interior, axes rotated 25 deg'),
                        ('ring_r35_lin', 'linear interior')):
        for cpl in ('directional', 'conormalbc', 'conormal', 'scalar'):
            H, E2, Ei = [], [], []
            for lev in (1, 2, 3, 4, 6):
                c = f'{name}_{cpl}_L{lev}'
                j = db.j(c)
                if not j:
                    continue
                rr = j['ring']
                rows.append((label, cpl, f"{j['cells']['aniso']}+{j['cells']['iso']}", e(j['error']['exact']['all']['L2'], 3),
                             e(j['error']['exact']['all']['Linf'], 3), g(j['interface']['imbalance'], 6),
                             g(rr['predicted_creation_C6'], 6) if cpl == 'directional' else '',
                             g(rr['gross_rate_into_iso'], 6), e(j.get('steady_change_since_previous_write'), 0)))
                H.append(1.0 / lev); E2.append(j['error']['exact']['all']['L2']); Ei.append(j['error']['exact']['all']['Linf'])
            if len(H) > 1:
                orders[(name, cpl)] = dict(L2=order(H, E2), Linf=order(H, Ei), levels=[round(1 / h) for h in H], L2err=E2, Linferr=Ei)
    t = md(['data', 'coupling', 'cells', 'L2 vs exact [K]', 'Linf vs exact [K]', 'created at interface [W/m]',
            '(kx-ky) pi a2 [W/m]', 'gross rate [W/m]', 'steady change [K]'], rows)
    rr = []
    for r in RATIOS:
        line = [r]
        for cpl in ('directional', 'conormalbc'):
            c = f'ring_r{r}_q_{cpl}_L2'
            if c in db.by and db.by[c]['status'] == 'aborted':
                line += ['diverged', 'diverged', '']
            else:
                line += [e(db.v(c, 'L2_exact'), 3), e(db.v(c, 'Linf_exact'), 3), g(db.v(c, 'imbalance'), 5)]
        rr.append(line)
    t2 = md(['kx/ky', 'directional: L2 [K]', 'Linf [K]', 'created [W/m]', 'corrected: L2 [K]', 'Linf [K]', 'created [W/m]'], rr)
    return t, t2, orders


# ------------------------------------------------------------------ P3


def p3_grid(db, cpl='directional', lev=2):
    rows = []
    for no in (1, 2, 3):
        for m in (1, 2, 4, 8):
            c = f'p3_{cpl}_o{no}_L{lev}_dt{m}'
            if c in db.by:
                dr, dE = db.v(c, 'energy_drift'), db.v(c, 'dE_aniso')
                rows.append((no, 0.4 / m, e(db.v(c, 'err_staggering'), 3), e(db.v(c, 'err_euler'), 3), e(db.v(c, 'err_space'), 3),
                             e(db.v(c, 'err_defect'), 3), g(db.v(c, 'energy_ratio_raw'), 9), e(dr, 3), e(abs(dr / dE), 3),
                             e(db.v(c, 'err_solver_minus_emulator'), 1)))
    return md(['passes', 'dt [s]', 'staggering [K]', 'implicit Euler [K]', 'space [K]', 'conservation defect [K]',
               'raw energy ratio', 'drift [J/m]', 'drift / exchanged', 'solver minus emulator [K]'], rows)


def p3_orders(db):
    """Observed temporal order of the staggering error and of the drift, and the contraction per pass."""
    out = []
    for cpl in ('directional', 'scalar'):
        for lev in (1, 2, 4, 8):
            for no in (1, 2, 3):
                S, Dr = [], []
                for m in (1, 2, 4, 8):
                    c = f'p3_{cpl}_o{no}_L{lev}_dt{m}'
                    S.append(db.v(c, 'err_staggering')); Dr.append(db.v(c, 'energy_drift'))
                if None in S:
                    continue
                oS = [math.log2(S[i] / S[i + 1]) for i in range(3)]
                oD = [math.log2(abs(Dr[i] / Dr[i + 1])) for i in range(3)]
                out.append((cpl, f'{16 * lev}+{24 * lev}', no, ' '.join(f'{x:.2f}' for x in oS), ' '.join(f'{x:.2f}' for x in oD)))
    t1 = md(['coupling', 'cells', 'passes', 'order of the staggering error (dt 0.4 > 0.2 > 0.1 > 0.05)',
             'order of the energy drift'], out)
    con = []
    for lev in (1, 2, 4, 8):
        for m in (1, 2, 4, 8):
            s = [db.v(f'p3_directional_o{no}_L{lev}_dt{m}', 'err_staggering') for no in (1, 2, 3)]
            d = [db.v(f'p3_directional_o{no}_L{lev}_dt{m}', 'energy_drift') for no in (1, 2, 3)]
            if None in s or None in d:
                continue
            con.append((f'{16 * lev}+{24 * lev}', 0.4 / m, e(s[0], 2), e(s[1], 2), e(s[2], 2), f'{s[1] / s[0]:.4f}', f'{s[2] / s[1]:.4f}',
                        f'{abs(d[1] / d[0]):.4f}', f'{abs(d[2] / d[1]):.4f}'))
    t2 = md(['cells', 'dt [s]', 'staggering, 1 pass [K]', '2 passes', '3 passes', 'factor 1 to 2', 'factor 2 to 3',
             'drift factor 1 to 2', 'drift factor 2 to 3'], con)
    return t1, t2


def p3_defect(db):
    rows = []
    for no in (1, 2, 3):
        for m in (1, 2, 4, 8):
            c = f'p3_scalar_o{no}_L2_dt{m}'
            if c in db.by:
                rows.append((no, 0.4 / m, e(db.v(c, 'err_defect'), 4), e(db.v(c, 'err_staggering'), 3),
                             g(db.v(c, 'energy_ratio_raw'), 9), g(db.v(c, 'energy_ratio_destaggered'), 12)))
    return md(['passes', 'dt [s]', 'conservation defect [K]', 'staggering [K]', 'raw energy ratio', 'staggering term removed'], rows)


def p3_space(db):
    rows = []
    for cpl in ('directional', 'scalar'):
        E, n = [], []
        for lev in (1, 2, 4, 8):
            c = f'p3_{cpl}_o3_L{lev}_dt8'
            v = db.v(c, 'err_space')
            if v is not None:
                E.append(v); n.append(lev)
        if len(E) > 1:
            rows.append((cpl, ' '.join(f'{x:.3e}' for x in E), ' '.join(f'{x:.2f}' for x in order([1 / l for l in n], E))))
    return md(['coupling', 'spatial error on 16+24, 32+48, 64+96, 128+192 cells [K]', 'observed order'], rows)


def p3_sweep(db):
    rows = []
    for r in (1, 10, 35, 200, 1000):
        for th in (0, 30, 60, 90):
            s, d = f'p3_r{r}_t{th}_scalar', f'p3_r{r}_t{th}_directional'
            if s in db.by:
                rows.append((r, th, g(db.v(s, 'ratio_predicted'), 9), g(db.v(s, 'energy_ratio_destaggered'), 12),
                             e(db.v(s, 'err_defect'), 3), e(db.v(s, 'Linf_scheme'), 2), g(db.v(d, 'energy_ratio_destaggered'), 13),
                             e(db.v(d, 'Linf_exact'), 2)))
    return md(['k1/k2', 'theta', 'predicted ratio', 'scalar: measured, staggering removed', 'scalar: conservation defect [K]',
               'scalar: Linf vs own problem [K]', 'directional: measured ratio', 'directional: Linf vs exact [K]'], rows)


def p3_sheared(db):
    rows = []
    for cid in ('p3s_directional_dt1', 'p3s_directional_dt2', 'p3s_directional_dt4', 'p3s_conormal_dt1', 'p3s_conormal_dt2',
                'p3s_conormal_dt4', 'p3s_conormal_dt8', 'p3s_conormalbc_dt1', 'p3s_conormalbc_dt2', 'p3s_conormalbc_dt4',
                'p3s_conormalbc_dt8', 'p3s_directional_dt2_o3', 'p3s_conormal_dt2_o3', 'p3s_conormalbc_dt2_o3'):
        r = db.by.get(cid)
        if r:
            rows.append((r['coupling'], r['dt'], r['nouter'], e(fl(r['Linf_exact']), 3), e(fl(r['Linf_scheme']), 3),
                         e(fl(r['err_solver_minus_emulator']), 1), e(fl(r['err_staggering']), 3), e(fl(r['err_euler']), 3)))
    return md(['coupling', 'dt [s]', 'passes', 'Linf vs exact series [K]', 'Linf vs series of own problem [K]',
               'solver minus emulator [K]', 'staggering [K]', 'implicit Euler [K]'], rows)


def variants(db):
    rows = []
    for name in ('rev', 'mode1', 'mode3', 'dd'):
        for m in (1, 2, 4):
            c = f'p3_{name}_dt{m}'
            r = db.by.get(c)
            if r:
                rows.append((name, 0.4 / m, r['order'], e(fl(r['Linf_exact']), 3), e(fl(r['err_staggering']), 3),
                             g(fl(r['energy_ratio_raw']), 8) if r['variant'].startswith('NN') else '',
                             e(fl(r['energy_drift']), 3) if r['variant'].startswith('NN') else '',
                             e(fl(r['err_solver_minus_emulator']), 1)))
    return md(['variant', 'dt [s]', 'solve order', 'Linf vs exact [K]', 'staggering [K]', 'raw energy ratio', 'drift [J/m]',
               'solver minus emulator [K]'], rows)


# ------------------------------------------------------------------ wallHeatFlux


def whf(db):
    rows = []
    cases = [('xaud_p1_t90_directional', 'patch test, 90 deg (aligned), directional'),
             ('xaud_p1_t0_directional', 'patch test, 0 deg (aligned), directional'),
             ('xaud_p1_t30_scalar', 'patch test, 30 deg, scalar'), ('xaud_p1_t30_directional', 'patch test, 30 deg, directional'),
             ('xaud_p1_t30_conormalbc', 'patch test, 30 deg, co-normal BC'),
             ('xaud_p2A_r35_scalar_L1', 'annulus A, scalar'), ('xaud_p2A_r35_directional_L1', 'annulus A, directional'),
             ('xaud_ring_r35_q_directional_L2', 'ring, directional'), ('xaud_ring_r35_q_conormalbc_L2', 'ring, co-normal BC')]
    for c, label in cases:
        j = db.j(c)
        if not j:
            continue
        i = j['interface']
        rows.append((label, g(i['into_iso'], 8), g(i.get('stock_wallHeatFlux_iso'), 8), g(i['into_aniso'], 8),
                     g(i.get('coNormalWallHeatFlux_aniso'), 8), g(i.get('stock_wallHeatFlux_aniso'), 8),
                     g(i['into_aniso_normal'], 8),
                     g(i['stock_wallHeatFlux_aniso'] / i['into_aniso_normal'], 9) if i.get('stock_wallHeatFlux_aniso') and i['into_aniso_normal'] else '',
                     e(i['stock_wallHeatFlux_aniso'] + i['stock_wallHeatFlux_iso'], 2) if i.get('stock_wallHeatFlux_aniso') is not None else '',
                     e(i['into_aniso'] + i['into_iso'], 2)))
    t = md(['case', 'into iso: post.py', 'into iso: stock object', 'into aniso: post.py (tensor)',
            'into aniso: coNormalWallHeatFlux', 'into aniso: stock object', 'into aniso, normal part only',
            'stock / normal part', 'stock two-sided sum', 'true two-sided sum'], rows)
    return t


def whf_transient(db):
    pj = json.load(open(os.path.join(BENCH, 'predictions_tested.json')))
    tab = next((p.get('table') for p in pj if p['id'] == 'P21'), None)
    if not tab:
        return 'not available'
    rows = [(r['case'], g(r['factor_predicted'], 10), g(r['stock_aniso_over_true_storage'], 10),
             g(r['stock_aniso_integral'], 9), g(r['stock_iso_integral'], 9), g(r['dE_aniso'], 9), g(r['dE_iso'], 9),
             e(r['books_stock'], 3), e(r['books_true'], 3), e(r['corrected_object_over_storage_aniso'] - 1, 1),
             e(r['stock_iso_over_storage_iso'] - 1, 1)) for r in tab]
    return md(['case', 'predicted factor', 'stock aniso book / true storage change', 'stock book aniso [J/m]', 'stock book iso [J/m]',
               'storage change aniso [J/m]', 'storage change iso [J/m]', 'sum of stock books', 'sum of storage changes',
               'corrected object / storage - 1 (aniso)', 'stock object / storage - 1 (iso)'], rows)


# ------------------------------------------------------------------ predictions


def predictions():
    pj = json.load(open(os.path.join(BENCH, 'predictions_tested.json')))

    def short(v, n=120):
        if v is None:
            return ''
        if isinstance(v, float):
            return f'{v:.10g}'
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            return ', '.join(f'{x:.9g}' for x in v)
        if isinstance(v, dict) and all(isinstance(x, (int, float)) for x in v.values()) and len(v) <= 6:
            return '; '.join(f'{k}: {x:.9g}' for k, x in v.items())
        s = json.dumps(v, default=float)
        return s if len(s) <= n else s[:n - 3] + '...'
    rows = [(p['id'], p['title'], '**' + p['verdict'] + '**', short(p['predicted']), short(p['measured']), p['note']) for p in pj]
    c = {}
    for p in pj:
        c[p['verdict']] = c.get(p['verdict'], 0) + 1
    return md(['id', 'prediction', 'verdict', 'predicted', 'measured here', 'how it was tested'], rows), c, pj


if __name__ == '__main__':
    db = DB()
    for name, fn in (('inventory', inventory), ('h1_ratio', h1_ratio), ('h1_other', h1_other), ('meshindep', meshindep),
                     ('p1_stats', p1_stats), ('p1_mesh', p1_mesh), ('frame', frame), ('annulus', annulus),
                     ('annulus_ratios', annulus_ratios), ('trap', trap), ('ring', ring), ('p3_grid', p3_grid),
                     ('p3_orders', p3_orders), ('p3_defect', p3_defect), ('p3_space', p3_space), ('p3_sweep', p3_sweep),
                     ('p3_sheared', p3_sheared), ('variants', variants), ('whf', whf), ('whf_transient', whf_transient)):
        print('=====', name)
        try:
            out = fn(db)
            for o in (out if isinstance(out, tuple) else (out,)):
                print(o if isinstance(o, str) else str(o)[:3000])
        except Exception as ex_:
            import traceback
            traceback.print_exc()
    for cpl in ('directional', 'scalar', 'conormalbc'):
        print('===== p1 grid', cpl)
        print(p1_grid(db, cpl))


# ------------------------------------------------------------------ stability of the corrected coupling


def stability(db):
    """Every run of block x_stab: outcome of the pseudo-time (or physical-time) march."""
    rows = []
    for pid, _, a in blocks.x_stab():
        f = os.path.join(db.cases, pid)
        ab = os.path.join(f, 'aborted.json')
        j = db.j(pid)
        P = (json.load(open(ab))['params'] if os.path.exists(ab) else (j['params'] if j else None))
        if P is None:
            continue
        cap = 'physical (2.0e6, 1.5e6)' if P['rhoA'] > 100 else 'pseudo (1e3, 1e3)'
        if os.path.exists(ab):
            out = f"diverged after {json.load(open(ab))['steps_completed']} steps"
        else:
            out = f"stable; error {j['error']['exact']['all']['Linf']:.1e} K after {j['log']['steps']} steps"
        rows.append((round(P['k1'] / P['k2']), P['theta'], P['coupling'], P['ki'], cap, P['dt'], P['init'], out))
    return md(['k1/k2', 'theta', 'coupling', 'k_i', 'rho c_p [J/m3/K]', 'dt [s]', 'start', 'outcome'], rows)


def diverged_list(db, pattern='conormal'):
    out = []
    for r in db.rows:
        if r['status'] == 'aborted' and pattern in r['coupling'] and not r['id'].startswith('xguard'):
            out.append(r['id'])
    return out
