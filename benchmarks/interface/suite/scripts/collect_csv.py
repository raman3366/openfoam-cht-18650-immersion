#!/usr/bin/env python3
"""
collect_csv.py -- one tidy table with one row per run.

    collect_csv.py [<casesDir> [<out.csv>]]        default: ../cases -> ../results.csv

Reads <case>/result.json (or aborted.json) of every case directory and <case>/bench.json.
Empty cell = quantity not defined for that problem.  Heat rates in W per metre depth,
energies in J per metre depth, temperatures in K.  Column meanings are listed in COLUMNS.
"""
import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blocks       # noqa: E402
import exact as ex  # noqa: E402

COLUMNS = [
    ('id', 'case directory name under cases/'),
    ('family', 'design = point of the design matrix (same id), added = added by the benchmark job'),
    ('block', 'design-matrix tag (P1-patch, P3-grid, ...) or block of blocks.py'),
    ('problem', 'patch (P1), annulus (P2), slab (P3/P4), ring (P5)'),
    ('coupling', 'scalar | directional | function | conormal (function object) | conormalbc (derived BC) | scalarbc'),
    ('status', 'ok | aborted'),
    ('ratio', 'k_high/k_low of the solid'),
    ('k_principal', 'principal conductivities as given to the solid'),
    ('theta_deg', 'angle of principal axis 1 from the interface normal (strips); psi for the ring'),
    ('variant', 'outer conditions / annulus variant / ring data'),
    ('cells_aniso', ''), ('cells_iso', ''), ('n_normal', 'cells across the anisotropic region'),
    ('dt', 'time step [s]'), ('nsteps', 'time steps run'), ('nouter', 'coupling passes per step'),
    ('order', 'solve order of the regions'),
    ('k_c', 'conductivity handed to the coupling by the anisotropic side'),
    ('k_nn', 'n.K.n'),
    ('ratio_predicted', 'k_c/(n.K.n)'),
    ('L2_exact', 'volume-weighted rms error against the exact solution'),
    ('Linf_exact', 'largest cell error against the exact solution'),
    ('Linf_scheme', 'largest cell error against the closed-form solution of the transmission problem the coupling enforces'),
    ('dT_scale', 'temperature scale of the problem'),
    ('rate_into_iso', 'heat rate into iso through the interface (iso stencil)'),
    ('rate_out_of_aniso', 'heat rate out of aniso through the interface (tensor stencil, normal + tangential part)'),
    ('rate_out_of_aniso_normal', 'normal part of the previous column'),
    ('rate_out_of_aniso_tangential', 'tangential part'),
    ('imbalance', 'rate_into_iso - rate_out_of_aniso: energy created at the interface per unit time'),
    ('imbalance_predicted', 'K_nt B H (strip, stock options), (kx-ky) pi a2 (ring, stock directional), 0 (corrected)'),
    ('rate_exact', 'exact heat rate through the interface in the aniso->iso direction'),
    ('ratio_steady', 'rate_into_iso / rate_out_of_aniso_normal (steady problems)'),
    ('energy_ratio_raw', 'slab: -dE_iso/dE_aniso over the run (contains the staggering term for one pass)'),
    ('energy_ratio_destaggered', 'slab, one pass: ratio with the staggering term of Eq. (E2) removed'),
    ('energy_ratio_emulator', 'slab: same raw ratio from the Python emulator of the algorithm'),
    ('energy_drift', 'slab: dE_aniso + dE_iso'),
    ('dE_aniso', ''), ('dE_iso', ''),
    ('staggering_term', 'slab, one pass: dt C_X H (T_Y1(end) - T_Y1(0))'),
    ('err_solver_minus_emulator', 'slab: Linf, solver against the emulator'),
    ('err_staggering', 'slab: Linf, emulator against monolithic implicit Euler'),
    ('err_euler', 'slab: Linf, monolithic implicit Euler against semi-discrete'),
    ('err_space', 'slab: Linf, semi-discrete against the series of the scheme problem'),
    ('err_defect', 'slab: Linf, series of the scheme problem against the exact series'),
    ('stock_whf_aniso', 'stock wallHeatFlux object, anisotropic side, integrated over the interface (end of run)'),
    ('stock_whf_iso', 'stock wallHeatFlux object, isotropic side'),
    ('cn_whf_aniso', 'coNormalWallHeatFlux object, anisotropic side'),
    ('cn_whf_iso', 'coNormalWallHeatFlux object, isotropic side'),
    ('valueFraction_diff', 'written valueFraction of the anisotropic side minus b_nbr/(b_own + b_nbr)'),
    ('Qp_inner', 'annulus: heat rate per radian at the inner wall'),
    ('Qp_out_of_aniso', 'annulus: per radian, out of aniso at the interface'),
    ('Qp_into_iso', 'annulus: per radian, into iso at the interface'),
    ('Qp_exact', 'annulus: exact value'),
    ('a2_first_cell', 'ring: cos(2 phi) coefficient of the first-solid-cell temperature'),
    ('gross_rate', 'ring: integral of |q| into iso over the interface (the net rate is zero)'),
    ('steady_change', 'steady problems: largest change between the last two written fields'),
    ('continuations', 'steady problems: number of times the driver doubled the run'),
    ('h_solves', 'number of "Solving for h" lines in the log(s)'),
    ('region_solves', 'number of "Solving for solid region" lines'),
    ('cpu_s', 'solver CPU time'),
    ('abort_reason', 'aborted runs: first lines of the fatal message'),
]


def tags():
    t = {}
    for p in blocks.M.points():
        t[p['id']] = ('design', p['tag'])
    for name in dir(blocks):
        if name.startswith('x_'):
            for pid, _, _ in getattr(blocks, name)():
                t.setdefault(pid, ('added', name))
    return t


def g(d, *keys, default=''):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def row(case, tagmap):
    cid = os.path.basename(case)
    fam, blk = tagmap.get(cid, ('added', ''))
    out = dict(id=cid, family=fam, block=blk)
    fr, fa = os.path.join(case, 'result.json'), os.path.join(case, 'aborted.json')
    if os.path.exists(fr):
        r = json.load(open(fr)); out['status'] = 'ok'
    elif os.path.exists(fa):
        r = json.load(open(fa)); out['status'] = 'aborted'
        out['abort_reason'] = ' '.join(r.get('fatal_message', '').split())[:160]
    else:
        return None
    P = r['params']
    out.update(problem=r['problem'], coupling=P['coupling'], dt=P['dt'], nsteps=P['nsteps'], nouter=P['nouter'],
               order=P['order'])
    if r['problem'] in ('patch', 'slab'):
        k = (P['k1'], P['k2'], P['k3'])
        out.update(ratio=max(k) / min(k), theta_deg=P['theta'], n_normal=P['nxa'])
        knn = ex.knn_knt(k, P['theta'])[0]
        out['variant'] = (('sheared' if P.get('ybc') == 'gradient' else 'dirichlet') if r['problem'] == 'patch'
                          else P['xbc'] + '/' + P['ic'] + ('/B' if P.get('B') else ''))
        if P.get('phi'):
            out['variant'] += f"/phi{P['phi']:g}"
    elif r['problem'] == 'annulus':
        k = (P['kr'], P['kth'], P['kz'])
        out.update(ratio=max(k) / min(k), n_normal=P['nra'], theta_deg=0.0)
        knn = P['kr']
        out['variant'] = ('A' if P['kth'] == P['kr'] else ('B' if P['sector'] == 360 else 'C')) + f"/sector{P['sector']:g}"
    else:
        k = (P['kx'], P['ky'], P['kz'])
        out.update(ratio=max(k) / min(k), n_normal=P['nra'], theta_deg=P['psi'])
        knn = ''
        out['variant'] = ('harm2' if P.get('bcdata') == 'harm2' else ('quadratic' if P['aq'] else 'linear'))
    out['k_principal'] = ' '.join(f'{v:g}' for v in k)
    kc = {'scalar': ex.frob(k), 'scalarbc': ex.frob(k), 'function': P.get('kc')}.get(P['coupling'], knn)
    out.update(k_c=kc, k_nn=knn, ratio_predicted=(kc / knn if knn not in ('', 0) and kc != '' else ''))
    if out['status'] == 'aborted':
        return out
    out.update(cells_aniso=r['cells']['aniso'], cells_iso=r['cells']['iso'], dT_scale=r['dT_scale'],
               L2_exact=g(r, 'error', 'exact', 'all', 'L2'), Linf_exact=g(r, 'error', 'exact', 'all', 'Linf'),
               Linf_scheme=g(r, 'error', 'scheme', 'all', 'Linf'))
    i = r['interface']
    if r['problem'] == 'slab':
        ia, ian, iat, ii = i['recomputed_into_aniso'], '', '', i['recomputed_into_iso']
    else:
        ia, ian, iat, ii = i['into_aniso'], i['into_aniso_normal'], i['into_aniso_tangential'], i['into_iso']
    out.update(rate_into_iso=ii, rate_out_of_aniso=-ia, imbalance=ii + ia,
               rate_out_of_aniso_normal=(-ian if ian != '' else ''),
               rate_out_of_aniso_tangential=(-iat if iat != '' else ''),
               stock_whf_aniso=i.get('stock_wallHeatFlux_aniso', ''), stock_whf_iso=i.get('stock_wallHeatFlux_iso', ''),
               cn_whf_aniso=i.get('coNormalWallHeatFlux_aniso', ''), cn_whf_iso=i.get('coNormalWallHeatFlux_iso', ''),
               valueFraction_diff=i.get('valueFraction_maxdiff', ''))
    if ian not in ('', 0) and r['problem'] != 'slab':
        out['ratio_steady'] = ii / (-ian) if ian != 0 else ''
    if r['problem'] == 'patch':
        out.update(imbalance_predicted=r['exact']['predicted_creation'], rate_exact=r['exact']['rate_plus_xi'])
    if r['problem'] == 'annulus':
        m = r['measured_Qp']
        out.update(Qp_inner=m['inner_wall'], Qp_out_of_aniso=m['interface_out_of_aniso'],
                   Qp_into_iso=m['interface_into_iso'], Qp_exact=r['exact']['Qp'],
                   imbalance_predicted=(0.0 if P['coupling'] != 'scalar' else ''))
    if r['problem'] == 'ring':
        rr = r['ring']
        out.update(a2_first_cell=rr['a2_first_cell'], gross_rate=rr['gross_rate_into_iso'], rate_exact=0.0,
                   imbalance_predicted=(0.0 if P['coupling'] in ('conormal', 'conormalbc') else
                                        (rr['predicted_creation_C6'] if P['coupling'] == 'directional' else '')))
    a = r.get('audit')
    if a:
        s = a.get('split_Linf_K', {})
        out.update(energy_ratio_raw=a.get('ratio_received_over_released', ''),
                   energy_ratio_destaggered=a.get('destaggered_ratio', ''),
                   energy_ratio_emulator=g(a, 'emulator', 'ratio'), energy_drift=a['drift'],
                   dE_aniso=a['dE_aniso'], dE_iso=a['dE_iso'], staggering_term=a.get('staggering_term_J_per_m', ''),
                   err_solver_minus_emulator=s.get('solver_minus_emulator', ''),
                   err_staggering=s.get('staggering__emulator_minus_monolithic', ''),
                   err_euler=s.get('euler__monolithic_minus_semidiscrete', ''),
                   err_space=s.get('space__semidiscrete_minus_series_scheme', ''),
                   err_defect=s.get('conservation_defect__series_scheme_minus_series_exact', ''))
    out.update(steady_change=r.get('steady_change_since_previous_write', ''),
               h_solves=g(r, 'log', 'h_solves'), region_solves=g(r, 'log', 'region_solves'),
               cpu_s=g(r, 'log', 'cpu_s'), continuations=(g(r, 'log', 'runs', default=1) or 1) - 1)
    return out


def main(cases, dest):
    tagmap = tags()
    rows = []
    for case in sorted(glob.glob(os.path.join(cases, '*'))):
        if os.path.isdir(case) and not os.path.basename(case).startswith('_'):
            r = row(case, tagmap)
            if r:
                rows.append(r)
    names = [c for c, _ in COLUMNS]
    with open(dest, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=names, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.12g}' if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.splitext(dest)[0] + '_columns.txt', 'w') as fh:
        for c, d in COLUMNS:
            fh.write(f'{c:32s} {d}\n')
    print(len(rows), 'rows ->', dest)
    return rows


if __name__ == '__main__':
    base = os.path.dirname(HERE)
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, 'cases'),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, 'results.csv'))
