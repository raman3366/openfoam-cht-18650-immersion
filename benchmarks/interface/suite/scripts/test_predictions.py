#!/usr/bin/env python3
"""
test_predictions.py -- test every entry of ../derivation/predictions.json against runs of
THIS job and write predictions_tested.json (verdict, predicted and measured numbers, cases).

    test_predictions.py [<casesDir>]

Verdicts: confirmed | refuted | undecided.  A prediction is "confirmed" only if a run of
this job measured the predicted quantity within the stated tolerance; "undecided" means no
run of this job can decide it (the reason is given).  The measured numbers come from
result.json of the named cases (written by post.py from the solver's fields).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact as ex      # noqa: E402
import ofio             # noqa: E402

BENCH = os.path.dirname(HERE)
PRED = os.path.join(BENCH, '..', 'derivation', 'predictions.json')
FROB = ex.frob((28.0, 0.8, 0.8))


class Cases:
    def __init__(self, root):
        self.root = root

    def has(self, cid):
        return os.path.exists(os.path.join(self.root, cid, 'result.json'))

    def r(self, cid):
        f = os.path.join(self.root, cid, 'result.json')
        return json.load(open(f)) if os.path.exists(f) else None

    def aborted(self, cid):
        f = os.path.join(self.root, cid, 'aborted.json')
        return json.load(open(f)) if os.path.exists(f) else None

    def path(self, cid):
        return os.path.join(self.root, cid)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def run(root):
    C = Cases(root)
    pred = {p['id']: p for p in json.load(open(PRED))['predictions']}
    out = []

    def add(pid, verdict, predicted, measured, note, cases):
        out.append(dict(id=pid, title=pred[pid]['title'], section=pred[pid]['section'],
                        status_in_predictions_json=pred[pid]['status'], verdict=verdict,
                        predicted=predicted, measured=measured, note=note, cases=cases))

    def need(pid, cids):
        miss = [c for c in cids if not C.has(c)]
        if miss:
            add(pid, 'undecided', None, None, 'cases not run (yet): ' + ', '.join(miss), cids)
            return False
        return True

    # ---------------------------------------------------------------- P01
    cid = 'xd_slab_scalar_t90'
    nvf, wvf = 0, 0.0
    for c_ in sorted(os.listdir(root)):
        r_ = C.r(c_)
        if r_ and 'valueFraction_maxdiff' in r_.get('interface', {}) and not r_['params'].get('Rlayer'):
            nvf += 1
            wvf = max(wvf, r_['interface']['valueFraction_maxdiff'])
    if need('P01', [cid]):
        i = C.r(cid)['interface']
        exp = pred['P01']['expected']['valueFraction_anisotropic_side']
        ok = rel(i['valueFraction_aniso_written'], exp) < 1e-10
        add('P01', 'confirmed' if ok else 'refuted', exp, i['valueFraction_aniso_written'],
            f"written valueFraction of the anisotropic side; relative difference {rel(i['valueFraction_aniso_written'], exp):.1e}; "
            f"it corresponds to kappa_BC = {0.5 / i['valueFraction_aniso_written'] - 0.5:.10f} W/m/K (|K| = {FROB:.10f}).  "
            f"Over all {nvf} runs of this job that wrote the fraction (every coupling mode, ratio and angle) its largest "
            f"difference from b_nbr/(b_own + b_nbr) is {wvf:.1e}", [cid])

    # ---------------------------------------------------------------- P02
    cid = 'xd_onestep_cpT'
    if need('P02', [cid]):
        r = C.r(cid)
        i = r['interface']
        f, fc = i['valueFraction_iso_written'], i['first_cell']
        pre = f * fc['Ta1_start'] + (1 - f) * fc['Ti1_start']
        post_ = f * fc['Ta1_start'] + (1 - f) * fc['Ti1_end']
        exp = pred['P02']['expected']['T_patch_isotropic_after_step_1']
        ok = abs(i['Tw_iso'] - exp) < 1e-9
        add('P02', 'confirmed' if ok else 'refuted', exp, i['Tw_iso'],
            f"patch temperature of the isotropic side after one step; pre-solve mixed value {pre:.9f} K, post-solve mixed value "
            f"{post_:.9f} K (own cell went from {fc['Ti1_start']:.4f} to {fc['Ti1_end']:.4f} K); written valueFraction {f:.12f}",
            [cid])

    # ---------------------------------------------------------------- P03
    cids = [f'xd_slab_scalar_t{t}' for t in (0, 30, 45, 60, 90)]
    if need('P03', cids):
        rows, worst, worst_raw = [], 0.0, 0.0
        for t, row in zip((0, 30, 45, 60, 90), pred['P03']['expected']['table']):
            a = C.r(f'xd_slab_scalar_t{t}')['audit']
            rows.append(dict(theta_deg=t, predicted=row['expected_ratio'], measured_destaggered=a['destaggered_ratio'],
                             measured_raw=a['ratio_received_over_released'], derivation_job_raw=row['openfoam_ratio_raw']))
            worst = max(worst, rel(a['destaggered_ratio'], row['expected_ratio']))
            worst_raw = max(worst_raw, rel(a['ratio_received_over_released'], row['openfoam_ratio_raw']))
        add('P03', 'confirmed' if worst < 1e-9 else 'refuted', [r_['predicted'] for r_ in rows],
            [r_['measured_destaggered'] for r_ in rows],
            f'storage ratio with the staggering term of Eq. (E2) removed; largest relative deviation from |K|/(n.K.n) {worst:.1e}; '
            f'the raw ratios reproduce the derivation job\'s own runs (different generator) to {worst_raw:.1e}', cids)
        out[-1]['table'] = rows

    # ---------------------------------------------------------------- P04
    cids = ['xd_slab_scalar_kkk', 'xd_slab_directional_kkk']
    if need('P04', cids):
        s = C.r(cids[0])['audit']['destaggered_ratio']
        d = C.r(cids[1])['audit']['destaggered_ratio']
        ok = rel(s, math.sqrt(3)) < 1e-9 and rel(d, 1.0) < 1e-9
        add('P04', 'confirmed' if ok else 'refuted', dict(scalar=math.sqrt(3), directional=1.0), dict(scalar=s, directional=d),
            'kappa (2 2 2) through constAnIso, de-staggered storage ratio; the design-matrix block P4-kkk repeats it for k = 0.5, 2, 50',
            cids)

    # ---------------------------------------------------------------- P05
    cids = ['xd_slab_scalar_t90_n5', 'xd_slab_scalar_t90', 'xd_slab_scalar_t90_n40', 'xd_slab_scalar_t90_dt0.04',
            'xd_slab_scalar_t90_dt0.0025', 'xd_slab_scalar_t90_kf0.15', 'xd_slab_scalar_t90_kf5.0',
            'xd_slab_scalar_t90_RL0.0001', 'xd_slab_scalar_t90_RL0.001']
    if need('P05', cids):
        exp = pred['P05']['expected']['ratio']
        m = {c: C.r(c)['audit']['destaggered_ratio'] for c in cids}
        raw = {c: C.r(c)['audit']['ratio_received_over_released'] for c in cids}
        worst = max(rel(v, exp) for v in m.values())
        add('P05', 'confirmed' if worst < 1e-8 else 'refuted', exp, m,
            f'de-staggered ratio for 5/10/40 cells, dt 0.04/0.01/0.0025 s, neighbour conductivity 0.15/0.5/5, and layer resistance '
            f'1e-4 and 1e-3 m2K/W (the layer was emulator-only in predictions.json; here it is an OpenFOAM run with '
            f'thicknessLayers/kappaLayers on both sides): largest relative deviation {worst:.1e}.  Raw one-pass ratios for comparison: '
            + ', '.join(f'{v:.6f}' for v in raw.values()), cids)

    # ---------------------------------------------------------------- P06
    cids = [f'xd_slab_directional_t{t}' for t in (0, 30, 45, 60, 90)]
    if need('P06', cids):
        m = [C.r(c)['audit']['destaggered_ratio'] for c in cids]
        raw = [C.r(c)['audit']['ratio_received_over_released'] for c in cids]
        theirs = [row['openfoam_ratio_raw'] for row in pred['P06']['evidence']['table']]
        worst = max(rel(v, 1.0) for v in m)
        wr = max(rel(a, b) for a, b in zip(raw, theirs))
        add('P06', 'confirmed' if worst < 1e-9 else 'refuted', 1.0, m,
            f'directional option on the slab at 0/30/45/60/90 degrees, de-staggered: largest deviation from 1 is {worst:.1e}; '
            f'raw ratios agree with the derivation job\'s runs to {wr:.1e}', cids)

    # ---------------------------------------------------------------- P07
    cids = [f'xd_slabfix_{c}_t{t}_n{n}' for c in ('scalar', 'directional') for t in (45, 90) for n in (10, 40)]
    if need('P07', cids):
        E = pred['P07']['expected']
        rows, worst = [], 0.0
        for c in ('scalar', 'directional'):
            for t in (45, 90):
                for n in (10, 40):
                    r = C.r(f'xd_slabfix_{c}_t{t}_n{n}')
                    i, H = r['interface'], r['params']['H']
                    Tw_exp = E['T_w_scalar'] if c == 'scalar' else E[f'T_w_directional_theta{t}']
                    row = dict(coupling=c, theta_deg=t, cells=n, Tw_predicted=Tw_exp, Tw_measured=i['Tw_aniso'],
                               q_iso_W_m2=-i['into_iso'] / H * -1.0, q_aniso_normal_W_m2=-i['into_aniso'] / H,
                               change_between_last_two_writes_K=_slab_change(C.path(f'xd_slabfix_{c}_t{t}_n{n}')))
                    rows.append(row)
                    worst = max(worst, abs(i['Tw_aniso'] - Tw_exp))
        qs = [r_ for r_ in rows if r_['coupling'] == 'scalar']
        q_ok = max(rel(abs(r_['q_iso_W_m2']), E['q_iso_scalar_W_m2']) for r_ in qs)
        qa90 = max(rel(abs(r_['q_aniso_normal_W_m2']), E['q_aniso_scalar_theta90_W_m2']) for r_ in qs if r_['theta_deg'] == 90)
        qa45 = max(rel(abs(r_['q_aniso_normal_W_m2']), E['q_aniso_scalar_theta45_W_m2']) for r_ in qs if r_['theta_deg'] == 45)
        ok = worst < 1e-7 and max(q_ok, qa90, qa45) < 1e-7
        add('P07', 'confirmed' if ok else 'refuted',
            dict(T_w_scalar=E['T_w_scalar'], q_iso_scalar=E['q_iso_scalar_W_m2'], q_aniso_scalar_90=E['q_aniso_scalar_theta90_W_m2'],
                 q_aniso_scalar_45=E['q_aniso_scalar_theta45_W_m2']),
            dict(T_w_scalar=[r_['Tw_measured'] for r_ in qs], max_abs_Tw_error_K=worst,
                 max_rel_flux_error=max(q_ok, qa90, qa45)),
            f'steady slab with fixed ends reached by time marching; wall temperature of the scalar option is the same on 10 and 40 '
            f'cells and at 45 and 90 degrees; largest deviation of any wall temperature from its prediction {worst:.1e} K; '
            f'flux deviations: iso {q_ok:.1e}, aniso at 90 deg {qa90:.1e}, at 45 deg {qa45:.1e} (relative)', cids)
        out[-1]['table'] = rows

    # ---------------------------------------------------------------- P08
    stag, worst, n = {}, 0.0, 0
    for cid in sorted(os.listdir(root)):
        r = C.r(cid)
        if not r or r['problem'] != 'slab':
            continue
        a, P = r.get('audit', {}), r['params']
        if a.get('destaggered_ratio') is None or P['xbc'] != 'NN' or P.get('allow_leak') or P.get('B'):
            continue
        n += 1
        worst = max(worst, rel(a['destaggered_ratio'], a['predicted_ratio_continuum']))
    for c, dt in (('xd_slab_scalar_t90_dt0.04', 0.04), ('xd_slab_scalar_t90', 0.01), ('xd_slab_scalar_t90_dt0.0025', 0.0025)):
        if C.has(c):
            r = C.r(c)
            stag[dt] = r['audit']['staggering_term_J_per_m'] / r['params']['H']
    if n:
        theirs = pred['P08']['evidence']['staggering_J_m2_by_dt']
        cmp_ = {str(dt): dict(measured_J_m2=v, derivation_job_J_m2=theirs.get(str(dt))) for dt, v in stag.items()}
        lin = (stag[0.04] / stag[0.01], stag[0.01] / stag[0.0025]) if len(stag) == 3 else None
        add('P08', 'confirmed' if worst < 1e-9 else 'refuted', 'identity (E2) to 1e-10; term proportional to dt', dict(
            max_relative_error_of_identity=worst, one_pass_closed_slab_runs=n, staggering_term_by_dt=cmp_,
            ratio_of_terms_for_dt_ratio_4=lin),
            f'Eq. (E2) solved for the flux ratio and compared with k_c/(n.K.n) in all {n} one-pass closed-slab runs of this job '
            f'(design blocks P3-grid, P3-sweep, P4-* and the replicas): largest relative error {worst:.1e}.  The term itself for dt = '
            f'0.04, 0.01, 0.0025 s scales by {lin[0]:.3f} and {lin[1]:.3f} per factor 4 in dt (not exactly 4, because the end state '
            f'of the first cell also depends on dt)' if lin else 'time-step series incomplete',
            ['all one-pass closed slabs'] + list(stag and ['xd_slab_scalar_t90_dt0.04', 'xd_slab_scalar_t90', 'xd_slab_scalar_t90_dt0.0025']))
    else:
        add('P08', 'undecided', None, None, 'no slab runs found', [])

    # ---------------------------------------------------------------- P09
    cids = ['xd_eq_scalar', 'xd_eq_directional']
    if need('P09', cids):
        E = pred['P09']['expected']
        m, spread = {}, {}
        for c, key in zip(cids, ('T_eq_scalar', 'T_eq_directional')):
            Tall = _all_T(C.path(c))
            m[key], spread[key] = float(np.mean(Tall)), float(np.ptp(Tall))
        # Eq. (E3) evaluated here from the case parameters (X = iso solved first, Y = aniso)
        mine = {}
        for c, key in zip(cids, ('T_eq_scalar', 'T_eq_directional')):
            P = C.r(c)['params']
            kc = FROB if key.endswith('scalar') else 0.8
            dxa, dxi = P['La'] / P['nxa'], P['Li'] / P['nxi']
            ba, ca, bi = kc / (dxa / 2), 0.8 / (dxa / 2), P['ki'] / (dxi / 2)
            MX, MY = P['rhoI'] * P['cpI'] * P['Li'], P['rhoA'] * P['cpA'] * P['La']
            rho, CX = ba / ca, bi * ba / (ba + bi)
            mine[key] = (MX * P['Ti0'] + (rho * MY + P['dt'] * CX) * P['Ta0']) / (MX + rho * MY + P['dt'] * CX)
        worst = max(abs(m[k_] - E[k_]) for k_ in m)
        ok = worst < 1e-7 and max(spread.values()) < 1e-7
        add('P09', 'confirmed' if ok else 'refuted', dict(E, E3_evaluated_here=mine), dict(m, spread_of_final_field_K=spread),
            f'closed adiabatic pair run to a uniform state (120 s instead of the 400 s of predictions.json; remaining spread of the '
            f'field {max(spread.values()):.1e} K); final temperature against Eq. (E3): largest difference {worst:.1e} K.  '
            f'The energy-conserving value is {E["T_eq_energy_conserving"]:.7f} K: the scalar option ends {m["T_eq_scalar"] - E["T_eq_energy_conserving"]:+.3f} K '
            f'and the directional option, with one pass per step, {m["T_eq_directional"] - E["T_eq_energy_conserving"]:+.4f} K from it', cids)

    # ---------------------------------------------------------------- P10
    cids = ['xd_slab_directional_t90', 'xd_slab_directional_t90_o2', 'xd_slab_directional_t90_o3', 'xd_slab_directional_t90_rev',
            'xd_slab_scalar_t90', 'xd_slab_scalar_t90_o2', 'xd_slab_scalar_t90_o3']
    if need('P10', cids):
        E = pred['P10']['expected']['emulator_raw_ratio']
        m = {k_: C.r(c)['audit']['ratio_received_over_released'] for k_, c in zip(('1', '2', '3', 'anisotropic_first'), cids[:4])}
        sc = [C.r(c)['audit']['ratio_received_over_released'] for c in cids[4:]]
        worst = max(rel(m[k_], E[k_]) for k_ in m)
        contr = ((m['2'] - 1) / (m['1'] - 1), (m['3'] - 1) / (m['2'] - 1))
        ok = worst < 1e-9
        add('P10', 'confirmed' if ok else 'refuted', E, dict(m, scalar_option_1_2_3_passes=sc, contraction_per_pass=contr),
            f'raw storage ratio, directional, 1/2/3 passes and reversed order: largest relative deviation from the predicted values '
            f'{worst:.1e}; deviation from 1 contracts by {contr[0]:.4f} and {contr[1]:.4f} per pass (lumped upper estimate in '
            f'predictions.json: {pred["P10"]["expected"]["lumped_contraction_factor_dt_0p01"]:.4f}); the scalar option gives '
            f'{sc[0]:.6f}, {sc[1]:.8f}, {sc[2]:.10f} for 1/2/3 passes, i.e. it tends to {FROB / 0.8:.9f}, not to 1', cids)

    # ---------------------------------------------------------------- P11
    cids = [f'p1_r35_t{t}_directional_n16' for t in (0, 15, 30, 45, 60, 75, 90)]
    if need('P11', cids):
        rows, wi, wn = [], 0.0, 0.0
        theirs = {row['theta_deg']: row for row in pred['P11']['expected']['table']}
        wt = 0.0
        for t in (0, 15, 30, 45, 60, 75, 90):
            r = C.r(f'p1_r35_t{t}_directional_n16')
            i = r['interface']
            row = dict(theta_deg=t, imbalance_predicted=r['exact']['predicted_creation'], imbalance_measured=i['imbalance'],
                       normal_parts_mismatch=i['imbalance_normal_parts'], Linf_error_K=r['error']['exact']['all']['Linf'],
                       into_iso=i['into_iso'], out_of_aniso=-i['into_aniso'], exact_rate=r['exact']['rate_plus_xi'])
            if t in theirs:
                row['derivation_job_Linf_error_K'] = theirs[t]['openfoam_max_T_error_K']
                row['derivation_job_received_by_iso'] = theirs[t]['openfoam_heat_received_by_isotropic_W_per_m']
                if theirs[t]['openfoam_max_T_error_K'] > 1e-6:
                    wt = max(wt, rel(row['Linf_error_K'], theirs[t]['openfoam_max_T_error_K']))
            rows.append(row)
            wi = max(wi, abs(i['imbalance'] - r['exact']['predicted_creation']))
            wn = max(wn, abs(i['imbalance_normal_parts']))
        fails = [r_['Linf_error_K'] for r_ in rows if r_['theta_deg'] not in (0, 90)]
        ok = wi < 1e-7 and wn < 1e-7 and min(fails) > 0.1
        add('P11', 'confirmed' if ok else 'refuted', 'imbalance = K_nt B H; normal parts balance; linear solution not reproduced',
            dict(max_abs_imbalance_minus_prediction_W_m=wi, max_abs_normal_parts_mismatch_W_m=wn,
                 Linf_error_K_by_angle={r_['theta_deg']: r_['Linf_error_K'] for r_ in rows}),
            f'all-Dirichlet patch test, ratio 35, 16 cells per side, seven angles: created energy equals K_nt B H to {wi:.1e} W/m, '
            f'the two normal parts balance to {wn:.1e} W/m, the error is {min(fails):.2f} to {max(fails):.2f} K at the five oblique '
            f'angles and round-off at 0 and 90 degrees; the temperature errors at 30/45/60 degrees agree with the derivation job\'s '
            f'independent case to {wt:.1e} (relative)', cids)
        out[-1]['table'] = rows

    # ---------------------------------------------------------------- P12
    cids = [f'p1_r35_t45_directional_n{n}' for n in (8, 16, 32, 64)]
    have = [c for c in cids if C.has(c)]
    if len(have) >= 3:
        e = {int(c.rsplit('n', 1)[1]): C.r(c)['error']['exact']['all']['Linf'] for c in have}
        th = {row['cells_per_side']: row['openfoam_max_T_error_K'] for row in pred['P12']['expected']['table']}
        worst = max(rel(e[n], th[n]) for n in th if n in e)
        vals = [e[n] for n in sorted(e)]
        ok = worst < 1e-8 and vals[-1] >= vals[0]
        add('P12', 'confirmed' if ok else 'refuted', th, e,
            f'largest temperature error at 45 degrees on {sorted(e)} cells per side: ' + ', '.join(f'{v:.4f}' for v in vals)
            + f' K; it does not fall under refinement (it rises slightly); agreement with the derivation job on 8/16/32: {worst:.1e}',
            have)
    else:
        add('P12', 'undecided', None, None, 'fewer than three meshes available', cids)

    # ---------------------------------------------------------------- P13
    rows, worstT, worstI = [], 0.0, 0.0
    for cpl in ('conormal', 'conormalbc'):
        for t in (0, 15, 30, 45, 60, 75, 90):
            for n in (8, 16, 32, 64):
                c = f'p1_r35_t{t}_{cpl}_n{n}'
                if C.has(c):
                    r = C.r(c)
                    rows.append(dict(case=c, Linf_error_K=r['error']['exact']['all']['Linf'], imbalance=r['interface']['imbalance'],
                                     steady_change=r.get('steady_change_since_previous_write')))
                    worstT = max(worstT, rows[-1]['Linf_error_K'])
                    worstI = max(worstI, abs(rows[-1]['imbalance']))
    if rows:
        ok = worstT < 1e-9 and worstI < 1e-8
        add('P13', 'confirmed' if ok else 'refuted', dict(max_T_error_K='< 1e-9', imbalance_W_per_m='< 1e-8'),
            dict(runs=len(rows), max_T_error_K=worstT, max_abs_imbalance_W_per_m=worstI),
            f'ratio 35, {len(rows)} runs (function-object variant of the derivation job and the derived boundary condition of this job; '
            f'angles 0 to 90 degrees; 8 to 64 cells per side): largest temperature error {worstT:.1e} K, largest imbalance {worstI:.1e} W/m. '
            f'This is the tensor of the prediction (28 : 0.8).  Outside it: at ratios 100, 200 and 1000 the pseudo-time march of the '
            f'corrected coupling diverged at several oblique angles with the design\'s step (RESULTS.md section 3.5), so "every angle" '
            f'does not extend to every ratio', [r_['case'] for r_ in rows])
    else:
        add('P13', 'undecided', None, None, 'no corrected-coupling patch runs yet', [])

    # ---------------------------------------------------------------- P14
    cid = 'xd_p14_shear_t45_B1000'
    if need('P14', [cid]):
        r = C.r(cid)
        P = r['params']
        meas = r['interface']['Tw_aniso'] - P['B'] * P['H'] / 2 - r['exact']['Tw']
        exp = pred['P14']['expected']['wall_T_error_K']
        add('P14', 'confirmed' if abs(meas - exp) < 1e-8 else 'refuted', exp, meas,
            f'sheared strip (tangential gradient imposed by fixedGradient on the eta-walls, so the closed form holds for any interface '
            f'height; 16+16 by 4 cells): mean wall temperature minus exact, difference from the closed form {abs(meas - exp):.1e} K; '
            f'cell field equals the closed-form scheme solution to {r["error"]["scheme"]["all"]["Linf"]:.1e} K.  In predictions.json this '
            f'was emulator-only; here it is an OpenFOAM run', [cid])

    # ---------------------------------------------------------------- P15
    cids = ['xd_p15_t45_AeqB_scalar', 'xd_p15_t45_AeqB_directional']
    if need('P15', cids):
        s = C.r(cids[0])['error']['exact']['all']['Linf']
        d = C.r(cids[1])['error']['exact']['all']['Linf']
        exp = pred['P15']['expected']['emulator_max_T_error_K']
        add('P15', 'confirmed' if rel(s, exp) < 1e-6 else 'refuted', exp, dict(scalar=s, directional=d),
            f'45 degrees, A = B = 1000 K/m, 16 cells per side: the scalar option is wrong by only {s:.4e} K while the documented '
            f'directional option is wrong by {d:.3f} K; relative difference from the emulator value {rel(s, exp):.1e}', cids)

    # ---------------------------------------------------------------- P16
    cids = [f'ring_r35_q_directional_L{lv}' for lv in (1, 2, 3, 4, 6)] + [f'ring_r35_harm2_directional_L{lv}' for lv in (1, 2)]
    have = [c for c in cids if C.has(c)]
    if len(have) >= 2:
        rows = []
        for c in have:
            r = C.r(c)
            rr = r['ring']
            rows.append(dict(case=c, cells_around=r['params']['nth'] * 4, imbalance_measured=r['interface']['imbalance'],
                             C6_with_measured_a2=rr['predicted_creation_C6'],
                             relative_difference=rel(r['interface']['imbalance'], rr['predicted_creation_C6']),
                             a2_first_cell=rr['a2_first_cell'], gross_rate=rr['gross_rate_into_iso']))
        q = [r_ for r_ in rows if '_q_' in r_['case']]
        hm = {r_['cells_around']: r_ for r_ in rows if 'harm2' in r_['case']}
        ratio = [r_['imbalance_measured'] / r_['C6_with_measured_a2'] for r_ in q]
        # same mesh, different boundary data: the ratio must be the same number (a geometric factor)
        indep = max((abs(r_['imbalance_measured'] / r_['C6_with_measured_a2']
                         - hm[r_['cells_around']]['imbalance_measured'] / hm[r_['cells_around']]['C6_with_measured_a2'])
                     for r_ in q if r_['cells_around'] in hm), default=None)
        ok = (max(abs(v - 1) for v in ratio) < 0.01 and (len(ratio) < 3 or abs(ratio[-1] - 1) < abs(ratio[1] - 1))
              and (indep is None or indep < 1e-6))
        add('P16', 'confirmed' if ok else 'refuted', '(kx - ky) pi a2, to the discretisation error of the circle',
            dict(measured_over_predicted_by_mesh={r_['cells_around']: v for r_, v in zip(q, ratio)},
                 same_ratio_for_other_boundary_data_abs_diff=indep,
                 runs={r_['case']: dict(measured=r_['imbalance_measured'], predicted=r_['C6_with_measured_a2']) for r_ in rows}),
            'full ring with Cartesian orthotropy 28 : 0.8 inside (problem P5, added by this job), stock directional option: energy '
            'created at the closed curved interface over (kx - ky) pi a2, a2 being the measured cos(2 phi) coefficient of the '
            'first-cell temperature: ' + ', '.join(f'{v:.5f}' for v in ratio) + ' on ' + ', '.join(str(r_['cells_around']) for r_ in q)
            + ' cells around the circle.  The ratio is a property of the mesh, not of the solution: the run with different boundary '
            'data (second-harmonic wall temperature) gives the same ratio on the same mesh'
            + (f' to {indep:.1e}' if indep is not None else '') + '.  It is the product of R over the first-cell-centre radius '
            '(above 1, first order in the cell size) and chord factors (below 1, second order in the angular step), so it tends to 1 '
            'from above after a maximum on the second mesh.  In predictions.json this was derived only', have)
        out[-1]['table'] = rows
    else:
        add('P16', 'undecided', None, None, 'ring runs not available yet', cids)

    # ---------------------------------------------------------------- P17
    cids = [f'xd_p17_box_t{t}_{c}' for t in (0, 45) for c in ('directional', 'conormalbc')]
    if need('P17', cids):
        rows = []
        for c in cids:
            r = C.r(c)
            P, a = r['params'], r['audit']
            Knt = ex.knn_knt((P['k1'], P['k2'], P['k3']), P['theta'])[1]
            T = r['time']
            fo = a.get('function_objects', {})
            rows.append(dict(case=c, theta_deg=P['theta'], coupling=P['coupling'], drift_J_per_m=a['drift'],
                             staggering_term_J_per_m=a.get('staggering_term_J_per_m'),
                             end_wall_leak_predicted_J_per_m=-Knt * P['B'] * P['H'] * T,
                             interface_creation_predicted_J_per_m=(0.0 if P['coupling'] == 'conormalbc' else Knt * P['B'] * P['H'] * T),
                             end_wall_rate_measured_W_per_m=r['patch_rates']['aniso']['xmin']['total'],
                             end_wall_rate_predicted_W_per_m=-Knt * P['B'] * P['H']))
        b = {(r_['theta_deg'], r_['coupling']): r_ for r_ in rows}
        leak = b[(45.0, 'conormalbc')]
        # one pass per step: drift = wall leak - staggering term, the latter from Eq. (E2) with the
        # first-cell temperature change of the region solved second (aniso)
        r = C.r('xd_p17_box_t45_conormalbc')
        P, fc = r['params'], r['interface']['first_cell']
        k = (P['k1'], P['k2'], P['k3'])
        da, di = P['La'] / P['nxa'] / 2, P['Li'] / P['nxi'] / 2
        ba, bi = ex.knn_knt(k, P['theta'])[0] / da, P['ki'] / di
        stag = P['dt'] * bi * ba / (ba + bi) * P['H'] * (fc['Ta1_end'] - fc['Ta1_start'])
        pred_drift = leak['end_wall_leak_predicted_J_per_m'] - stag
        ok = rel(leak['drift_J_per_m'], pred_drift) < 1e-8 and \
            abs(leak['end_wall_rate_measured_W_per_m'] - leak['end_wall_rate_predicted_W_per_m']) < 1e-8
        add('P17', 'confirmed' if ok else 'refuted',
            dict(end_wall_rate_W_per_m=leak['end_wall_rate_predicted_W_per_m'],
                 wall_leak_J_per_m=leak['end_wall_leak_predicted_J_per_m'], staggering_term_J_per_m=stag,
                 drift_J_per_m=pred_drift),
            dict(end_wall_rate_W_per_m=leak['end_wall_rate_measured_W_per_m'], drift_J_per_m=leak['drift_J_per_m'],
                 drift_stock_directional_45deg_J_per_m=b[(45.0, 'directional')]['drift_J_per_m'],
                 drift_aligned_J_per_m=b[(0.0, 'conormalbc')]['drift_J_per_m']),
            'closed strip (zero-gradient ends) with a tangential gradient of 1000 K/m, 20 s (design generator with the refusal of '
            'this combination switched off).  At 45 degrees with the corrected coupling, whose interface is conservative, the box '
            f'loses {-leak["drift_J_per_m"]:.4f} J/m; predicted: wall leak K_nt B H t = {-leak["end_wall_leak_predicted_J_per_m"]:.1f} '
            f'J/m minus the one-pass staggering term ({-stag:.4f} J/m, the first solid cell cooled by '
            f'{fc["Ta1_start"] - fc["Ta1_end"]:.2f} K) = {-pred_drift:.4f} J/m; relative difference '
            f'{rel(leak["drift_J_per_m"], pred_drift):.1e}.  The tensor-correct rate through the zero-gradient end wall is '
            f'{leak["end_wall_rate_measured_W_per_m"]:.6f} W/m (predicted {leak["end_wall_rate_predicted_W_per_m"]:.6f}).  With the '
            f'stock directional option the drift is {b[(45.0, "directional")]["drift_J_per_m"]:.1e} J/m: the energy it creates at the '
            'interface equals the energy lost through the wall, so a closed-box audit cannot see either.  Aligned control: drift '
            f'{b[(0.0, "conormalbc")]["drift_J_per_m"]:.1e} J/m', cids)
        out[-1]['table'] = rows

    # ---------------------------------------------------------------- P18
    cids = [f'xd_slab_cpT_n{n}' for n in (5, 10, 20, 40)]
    if need('P18', cids):
        rows, worst = [], 0.0
        for c, row in zip(cids, pred['P18']['expected']['table']):
            r = C.r(c)
            a = r['audit']
            fo = a.get('function_objects', {})
            inst = None
            if fo.get('stock_iso') and fo.get('conormal_aniso_total'):
                inst = fo['stock_iso']['last'] / (-fo['conormal_aniso_total']['last'])
            rows.append(dict(cells=row['cells'], energy_ratio_measured=a['ratio_received_over_released'],
                             energy_ratio_derivation_job=row['openfoam_energy_ratio'],
                             instantaneous_flux_ratio_last_step=inst, formula_D1=a.get('cp_factor_predicted_D1'),
                             eps_D2=a.get('cp_eps_predicted_D2')))
            worst = max(worst, rel(a['ratio_received_over_released'], row['openfoam_energy_ratio']))
        dev = [1 - r_['energy_ratio_measured'] for r_ in rows]
        inst_err = max(abs(r_['instantaneous_flux_ratio_last_step'] - r_['formula_D1']) for r_ in rows
                       if r_['instantaneous_flux_ratio_last_step'] is not None) if rows[0]['instantaneous_flux_ratio_last_step'] else None
        ok = worst < 1e-8
        add('P18', 'confirmed' if ok else 'refuted', [row['openfoam_energy_ratio'] for row in pred['P18']['expected']['table']],
            [r_['energy_ratio_measured'] for r_ in rows],
            f'Cp = 824.66 + 3.6 T in the isotropic region, directional coupling, four passes: energy ratio on 5/10/20/40 cells agrees '
            f'with the derivation job to {worst:.1e}; 1 - ratio = ' + ', '.join(f'{v:.5f}' for v in dev)
            + ' (falls by ' + ', '.join(f'{a_ / b_:.2f}' for a_, b_ in zip(dev[:-1], dev[1:])) + ' per halving of the spacing).  '
            + (f'New here: the per-step formula (D1) was checked against the SOLVER (it was emulator-only): flux into iso from the '
               f'stock object over flux out of aniso from the corrected object at the last step is '
               + ', '.join(f'{r_["instantaneous_flux_ratio_last_step"]:.7f}' for r_ in rows)
               + f', equal to mean Cp over Cp(T_wall) to {inst_err:.1e}; its deviation from 1 falls by '
               + ', '.join(f'{(1 - a_["instantaneous_flux_ratio_last_step"]) / (1 - b_["instantaneous_flux_ratio_last_step"]):.2f}'
                           for a_, b_ in zip(rows[:-1], rows[1:])) + ' per halving (first order)' if inst_err is not None else ''), cids)
        out[-1]['table'] = rows

    # ---------------------------------------------------------------- P19, P20, P24
    for pid in ('P19', 'P20', 'P24'):
        add(pid, 'undecided', pred[pid]['expected'], None,
            'needs a run of the 18650 fluid case, which this job was not allowed to make; the mechanism behind it is confirmed on the '
            'slab (' + {'P19': 'P18', 'P20': 'P08 and P10', 'P24': 'P03 and P21'}[pid] + ')', [])

    # ---------------------------------------------------------------- P21, P22
    cids = [f'xd_aud_{c}_t{t}' for c in ('scalar', 'directional') for t in (45, 90)] + ['xd_aud_scalar_kkk']
    if need('P21', cids):
        rows, w21, w22 = [], 0.0, 0.0
        for c in cids:
            r = C.r(c)
            a, i = r['audit'], r['interface']
            fo = a['function_objects']
            fac = a['predicted_ratio_continuum'] if r['params']['coupling'] == 'scalar' else \
                ex.frob((r['params']['k1'], r['params']['k2'], r['params']['k3'])) / r['E_knn'] if 'E_knn' in r else None
            k = (r['params']['k1'], r['params']['k2'], r['params']['k3'])
            fac = ex.frob(k) / ex.knn_knt(k, r['params']['theta'])[0]
            row = dict(case=c, factor_predicted=fac,
                       stock_aniso_over_true_storage=fo['stock_aniso']['integral'] / a['dE_aniso'],
                       stock_aniso_over_corrected_object=fo['stock_aniso']['integral'] / fo['conormal_aniso_total']['integral'],
                       corrected_object_over_storage_aniso=fo['conormal_aniso_total']['integral'] / a['dE_aniso'],
                       stock_iso_over_storage_iso=fo['stock_iso']['integral'] / a['dE_iso'],
                       books_stock=fo['stock_aniso']['integral'] + fo['stock_iso']['integral'],
                       books_true=a['dE_aniso'] + a['dE_iso'], dE_iso=a['dE_iso'], dE_aniso=a['dE_aniso'],
                       stock_aniso_integral=fo['stock_aniso']['integral'], stock_iso_integral=fo['stock_iso']['integral'])
            rows.append(row)
            w21 = max(w21, rel(row['stock_aniso_over_true_storage'], fac))
            w22 = max(w22, rel(row['stock_iso_over_storage_iso'], 1.0), rel(row['corrected_object_over_storage_aniso'], 1.0))
        add('P21', 'confirmed' if w21 < 1e-9 else 'refuted', {r_['case']: r_['factor_predicted'] for r_ in rows},
            {r_['case']: r_['stock_aniso_over_true_storage'] for r_ in rows},
            f'time integral of the stock wallHeatFlux object on the anisotropic side over that region\'s storage change: equals '
            f'|K|/(n.K.n) to {w21:.1e} for both coupling options.  With the scalar option the two stock books differ only by the '
            f'staggering term although the regions exchange unequal energy; with the directional option they are open by the factor',
            cids)
        out[-1]['table'] = rows
        add('P22', 'confirmed' if w22 < 1e-9 else 'refuted', '< 1e-10 relative', w22,
            f'time integral of the stock object on the isotropic side over that region\'s storage change, and of the corrected '
            f'object (coNormalWallHeatFlux of this job) on the anisotropic side over that region\'s: largest deviation from 1 is {w22:.1e}',
            cids)
    else:
        need('P22', cids)

    # ---------------------------------------------------------------- P23
    g = {c: (C.aborted(c), C.r(c)) for c in ('xguard_bc_scalar', 'xguard_fo_strict', 'xguard_fo_error', 'xguard_fo_warn',
                                             'xguard_fo_directional', 'xguard_fo_kkk')}
    if all(a or r for a, r in g.values()):
        desc = {c: (f"stopped by the guard after {a['steps_completed']} time-step headers" if a and a.get('guard_fired')
                    else ('stopped, not by the guard' if a else 'ran to the end')) for c, (a, r) in g.items()}
        ok = (all(g[c][0] and g[c][0]['guard_fired'] for c in ('xguard_bc_scalar', 'xguard_fo_strict', 'xguard_fo_error',
                                                                'xguard_fo_kkk'))
              and g['xguard_fo_warn'][1] is not None and g['xguard_fo_directional'][1] is not None)
        add('P23', 'confirmed' if ok else 'refuted', 'fatal error before the first step; warning only if not strict', desc,
            'guards of this job\'s library: the derived boundary condition stops at its first evaluation (inside the first step, '
            'before any solve); the function object stops the run when the function objects are loaded, before the first step, with '
            'its default settings and with "errors strict", and only warns with "action warn"; it does not fire for the directional '
            'option; it fires for (k k k) through constAnIso.  (The derivation job\'s own guard needs "errors strict" to stop a run; '
            'that was its stated behaviour and is not re-tested here)', list(g))
    else:
        add('P23', 'undecided', None, None, 'guard runs incomplete', list(g))

    # ---------------------------------------------------------------- P25
    cid = 'xd_p25_nnonorth2'
    if need('P25', [cid]):
        lg = C.r(cid)['log']
        per = lg['h_solves'] / lg['region_solves']
        add('P25', 'confirmed' if per == 1.0 else 'refuted', 1, per,
            f'nNonOrthogonalCorrectors 2 requested for the anisotropic region: {lg["h_solves"]} h solves in {lg["region_solves"]} '
            f'region solves over {lg["steps"]} steps, i.e. one assembly per region per pass', [cid])

    # ---------------------------------------------------------------- P26
    smoke = os.path.join(BENCH, 'smoke')
    obs = {}
    for c in ('x_expl_dir_eqcp', 'x_impl_dir_eqcp', 'x_impl_dir_cp15', 'x_impl_dir_cp15_o10', 'x_impl_sca_eqcp'):
        f = os.path.join(smoke, c, 'result.json')
        if os.path.exists(f):
            a = json.load(open(f))['audit']
            obs[c] = dict(energy_ratio=a['ratio_received_over_released'], drift_J_per_m=a['drift'],
                          solver_minus_monolithic_K=a['split_Linf_K']['staggering__emulator_minus_monolithic']
                          if False else None)
    add('P26', 'undecided', 'not predicted', obs,
        'predictions.json makes no prediction.  Observation from the re-run probes of the design job (smoke/x_impl_*): with '
        'useImplicit and equal Cp the directional option conserves energy to round-off; with unequal Cp (1000 and 1500 J/kg/K) the '
        'isotropic region receives 0.04 of what the anisotropic region releases, with one and with ten passes; with the scalar '
        'option the run diverges.  The cause was not diagnosed here', list(obs))

    out.sort(key=lambda d: d['id'])
    return out


def _all_T(case):
    tv, tn = ofio.time_dirs(case, 'aniso')[-1]
    vals = []
    for r in ('aniso', 'iso'):
        m = ofio.Mesh(os.path.join(case, 'constant', r, 'polyMesh'))
        vals.append(ofio.read_field(os.path.join(case, tn, r, 'T'), m.nCells, {k: v['n'] for k, v in m.patches.items()})[0])
    return np.concatenate(vals)


def _slab_change(case):
    t = ofio.time_dirs(case, 'aniso')
    if len(t) < 2:
        return None
    d = 0.0
    for r in ('aniso', 'iso'):
        m = ofio.Mesh(os.path.join(case, 'constant', r, 'polyMesh'))
        sz = {k: v['n'] for k, v in m.patches.items()}
        a = ofio.read_field(os.path.join(case, t[-1][1], r, 'T'), m.nCells, sz)[0]
        b = ofio.read_field(os.path.join(case, t[-2][1], r, 'T'), m.nCells, sz)[0]
        d = max(d, float(np.max(np.abs(a - b))))
    return d


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BENCH, 'cases')
    res = run(root)
    json.dump(res, open(os.path.join(BENCH, 'predictions_tested.json'), 'w'), indent=1, default=float)
    for d in res:
        print(f"{d['id']} {d['verdict']:10s} {d['title'][:70]}")
    c = {}
    for d in res:
        c[d['verdict']] = c.get(d['verdict'], 0) + 1
    print(c)
