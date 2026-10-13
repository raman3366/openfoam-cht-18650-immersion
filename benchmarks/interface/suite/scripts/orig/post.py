#!/usr/bin/env python3
"""
post.py -- measures for one benchmark case, computed from the written fields.

    post.py <caseDir>            writes <caseDir>/result.json and prints a summary

What is computed (DESIGN.md section 5):
  * temperature error against the exact solution and against the "scheme" solution
    (the transmission problem the selected coupling enforces): L2 (volume weighted),
    Linf, per region;
  * two-sided interface heat rates evaluated with the operators the solver uses:
        iso    :  sum |S| ki (Tw - Tc) delta
        aniso  :  sum |S| [ (n.K.n)(Tw - Tc) delta + (K n - (n.K.n) n) . grad(T)_patch ]
    with K the cell tensor (rotated at the cell centre, as heSolidThermo.C:188-190 does)
    and grad(T)_patch from the solver's own `grad` function object;
  * per-region steady audit (sum of the heat rates through ALL patches of a region);
  * what the stock wallHeatFlux object reports on both sides;
  * the coupling coefficient actually used (written valueFraction) against b_nbr/(b_own+b_nbr);
  * transient cases: energy of each region at every written time, exchange ratio,
    drift, and the split  solver | partitioned emulator | monolithic implicit Euler |
    semi-discrete | series(scheme) | series(exact).
All heat rates are in W per metre depth, energies in J per metre depth.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact as ex          # noqa: E402
import ofio                 # noqa: E402
import gencase              # noqa: E402
import fv1d                 # noqa: E402


def jf(x):
    """json-friendly float"""
    if isinstance(x, (np.ndarray, list, tuple)):
        return [jf(v) for v in x]
    return float(x)


class Case:
    def __init__(self, case):
        self.case = case
        self.P = json.load(open(os.path.join(case, 'bench.json')))
        self.E = gencase.exact_objects(self.P)
        self.mesh = {r: ofio.Mesh(os.path.join(case, 'constant', r, 'polyMesh')) for r in ('aniso', 'iso')}
        self.times = ofio.time_dirs(case, 'aniso')
        P = self.P
        if P['problem'] == 'annulus':
            self.kvec = (P['kr'], P['kth'], P['kz'])
            C = self.mesh['aniso'].C
            er = np.column_stack((C[:, 0], C[:, 1], 0 * C[:, 0]))
            er /= np.linalg.norm(er, axis=1)[:, None]
            ez = np.array([0.0, 0.0, 1.0])
            et = np.cross(ez, er)
            self.K = (P['kr'] * np.einsum('ci,cj->cij', er, er) + P['kth'] * np.einsum('ci,cj->cij', et, et)
                      + P['kz'] * np.einsum('i,j->ij', ez, ez)[None])
        else:
            self.kvec = (P['k1'], P['k2'], P['k3'])
            self.n, self.t = gencase.frame(P)
            K = ex.K_tensor(self.kvec, P['theta'] + P['phi'])
            self.K = np.tile(K, (self.mesh['aniso'].nCells, 1, 1))
        self.rhocp = dict(aniso=P['rhoA'] * P['cpA'], iso=P['rhoI'] * P['cpI'])

    # ------------------------------------------------------------ field access
    def field(self, tname, region, name):
        m = self.mesh[region]
        sizes = {k: v['n'] for k, v in m.patches.items()}
        path = os.path.join(self.case, tname, region, name)
        return ofio.read_field(path, m.nCells, sizes) if os.path.exists(path) else (None, None)

    def coords(self, X):
        """Interface coordinates of points: (xi, eta) for strips, (r, angle) for the annulus."""
        if self.P['problem'] == 'annulus':
            return np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
        return X @ self.n, X @ self.t

    # ------------------------------------------------------------ fluxes
    def patch_rates(self, region, T, Tb, gradTb):
        """Heat rate INTO the region through each of its patches [W per m depth]."""
        m, out = self.mesh[region], {}
        for pname, pp in m.patches.items():
            if pp['type'] == 'empty':
                continue
            g = m.patch(pname)
            Tc = T[g['cells']]
            ent = Tb.get(pname, {})
            Tw = np.broadcast_to(ent['value'], Tc.shape) if 'value' in ent else Tc
            sn = (Tw - Tc) * g['delta']
            if region == 'iso':
                qn, qt = self.P['ki'] * sn, np.zeros_like(sn)
            else:
                Kn = np.einsum('fij,fj->fi', self.K[g['cells']], g['n'])
                knn = np.einsum('fi,fi->f', Kn, g['n'])
                qn = knn * sn
                Kt = Kn - knn[:, None] * g['n']
                gb = gradTb.get(pname, {}).get('value') if gradTb else None
                qt = np.einsum('fi,fi->f', Kt, np.broadcast_to(gb, Kt.shape)) if gb is not None else np.zeros_like(sn)
            dz = self.P['dz']
            out[pname] = dict(normal=float(np.sum(g['magSf'] * qn) / dz),
                              tangential=float(np.sum(g['magSf'] * qt) / dz),
                              total=float(np.sum(g['magSf'] * (qn + qt)) / dz),
                              Tw_mean=float(np.sum(g['magSf'] * Tw) / np.sum(g['magSf'])))
        return out

    def energy(self, region, T):
        return float(self.rhocp[region] * np.sum(T * self.mesh[region].V) / self.P['dz'])

    def norms(self, region, err):
        V = self.mesh[region].V
        return dict(L2=float(math.sqrt(np.sum(V * err ** 2) / np.sum(V))), Linf=float(np.max(np.abs(err))))


def kc_face(c, g):
    """Conductivity the aniso side hands to the coupling, per interface face."""
    P = c.P
    Kn = np.einsum('fij,fj->fi', c.K[g['cells']], g['n'])
    knn = np.einsum('fi,fi->f', Kn, g['n'])
    if P['coupling'] == 'scalar':
        return np.full_like(knn, ex.frob(c.kvec)), knn
    if P['coupling'] == 'function':
        return np.full_like(knn, P['kc']), knn
    return knn, knn


def analyse(case):
    c = Case(case)
    P, E = c.P, c.E
    res = dict(case=os.path.abspath(case), problem=P['problem'], params=P)
    tval, tname = c.times[-1]
    T, Tb, G = {}, {}, {}
    for r in ('aniso', 'iso'):
        T[r], Tb[r] = c.field(tname, r, 'T')
    _, gb = c.field(tname, 'aniso', 'gradT')
    res['time'] = tval
    res['cells'] = {r: c.mesh[r].nCells for r in T}

    # ---------------------------------------------------------- exact fields
    def exact_T(region, which='exact', t=None):
        X = c.mesh[region].C
        a, b = c.coords(X)
        if P['problem'] == 'annulus':
            return E['obj'].T(E[which], a)
        if P['problem'] == 'patch':
            if which == 'scheme' and P['ybc'] != 'gradient':
                return None          # closed form of the scheme solution only for the sheared strip
            return E['obj'].T(E[which], a, b)
        obj, coef = (E['obj'], E['coef']) if which == 'exact' else (E['scheme_obj'], E['scheme_coef'])
        if coef is None:
            return None
        return obj.T(coef, a, tval if t is None else t) + P['B'] * b

    dT_scale = {'annulus': abs(P.get('T1', 0) - P.get('T3', 0)),
                'patch': abs(P.get('A', 0)) * P.get('La', 0) + abs(E['exact']['Ai']) * P['Li'] if P['problem'] == 'patch' else 0,
                'slab': max(abs(P.get('Ta0', 0) - P.get('Ti0', 0)), abs(P.get('TL', 0) - P.get('TR', 0))
                            if 'D' in P.get('xbc', '') else 0.0, P.get('mode_amp', 0) if P.get('ic') == 'mode' else 0.0)
                }[P['problem']]
    res['dT_scale'] = float(dT_scale)
    res['error'] = {}
    for which in ('exact', 'scheme'):
        tot = {}
        for r in T:
            Te = exact_T(r, which)
            if Te is None:
                continue
            tot[r] = c.norms(r, T[r] - Te)
        if tot:
            V = {r: c.mesh[r].V.sum() for r in tot}
            tot['all'] = dict(L2=float(math.sqrt(sum(V[r] * tot[r]['L2'] ** 2 for r in V) / sum(V.values()))),
                              Linf=float(max(tot[r]['Linf'] for r in V)))
            res['error'][which] = tot

    # ---------------------------------------------------------- interface, from written fields
    rates = {r: c.patch_rates(r, T[r], Tb[r], gb if r == 'aniso' else None) for r in T}
    res['patch_rates'] = rates
    ia, ii = rates['aniso']['aniso_to_iso'], rates['iso']['iso_to_aniso']
    inter = dict(into_aniso_normal=ia['normal'], into_aniso_tangential=ia['tangential'],
                 into_aniso=ia['total'], into_iso=ii['total'],
                 imbalance=ia['total'] + ii['total'],                 # energy created at the interface
                 imbalance_normal_parts=ia['normal'] + ii['total'],
                 Tw_aniso=ia['Tw_mean'], Tw_iso=ii['Tw_mean'])
    # consistent instantaneous evaluation from the two first-cell temperatures
    ga, gi = c.mesh['aniso'].patch('aniso_to_iso'), c.mesh['iso'].patch('iso_to_aniso')
    from scipy.spatial import cKDTree
    dist, j = cKDTree(gi['Cf']).query(ga['Cf'])
    assert dist.max() < 1e-9 * max(1.0, np.abs(ga['Cf']).max()) + 1e-12, 'interface faces do not match'
    kcf, knn = kc_face(c, ga)
    ba, ca, bi = kcf * ga['delta'], knn * ga['delta'], P['ki'] * gi['delta'][j]
    Ta1, Ti1 = T['aniso'][ga['cells']], T['iso'][gi['cells']][j]
    qtan = np.zeros_like(Ta1)
    if gb and 'aniso_to_iso' in gb and 'value' in gb['aniso_to_iso']:
        Kn = np.einsum('fij,fj->fi', c.K[ga['cells']], ga['n'])
        qtan = np.einsum('fi,fi->f', Kn - knn[:, None] * ga['n'], np.broadcast_to(gb['aniso_to_iso']['value'], Kn.shape))
    qr = -qtan if P['coupling'] == 'conormal' else 0.0
    Tw = (ba * Ta1 + bi * Ti1 + qr) / (ba + bi)
    S = ga['magSf'] / P['dz']
    inter.update(recomputed_into_aniso=float(np.sum(S * (ca * (Tw - Ta1) + qtan))),
                 recomputed_into_iso=float(np.sum(S * bi * (Tw - Ti1))),
                 recomputed_Tw=float(np.sum(S * Tw) / np.sum(S)))
    inter['recomputed_imbalance'] = inter['recomputed_into_aniso'] + inter['recomputed_into_iso']
    # coupling coefficient actually used by the solver
    fa_pred = bi / (ba + bi)
    ent = Tb['aniso']['aniso_to_iso']
    if 'valueFraction' in ent:
        inter['valueFraction_aniso_written'] = float(np.mean(ent['valueFraction']))
        inter['valueFraction_aniso_predicted'] = float(np.mean(fa_pred))
        inter['valueFraction_maxdiff'] = float(np.max(np.abs(ent['valueFraction'] - fa_pred)))
    # stock function objects
    for r, pn in (('aniso', 'aniso_to_iso'), ('iso', 'iso_to_aniso')):
        _, wb = c.field(tname, r, 'wallHeatFlux')
        if wb and pn in wb and 'value' in wb[pn]:
            g = c.mesh[r].patch(pn)
            inter[f'stock_wallHeatFlux_{r}'] = float(np.sum(g['magSf'] * wb[pn]['value']) / P['dz'])
    res['interface'] = inter
    res['region_audit'] = {r: float(sum(v['total'] for v in rates[r].values())) for r in rates}

    # ---------------------------------------------------------- problem-specific
    if P['problem'] == 'patch':
        s = E['obj']
        Hh = P['H']
        res['exact'] = dict(Tw=E['exact']['Tw'], rate_plus_xi=E['exact']['q_iso'] * Hh,
                            Knn=s.Knn, Knt=s.Knt, kc=E['kc'],
                            predicted_creation=(0.0 if P['coupling'] == 'conormal' else s.Knt * P['B'] * Hh),
                            scheme_Tw_sheared=E['scheme']['Tw'],
                            scheme_wall_error_sheared=E['scheme']['Tw'] - E['exact']['Tw'])
    if P['problem'] == 'annulus':
        ang = math.radians(P['sector'])
        res['exact'] = dict(Qp=E['exact']['Qp_in'], T2=E['exact']['T2'], kc=E['kc'], knn=E['knn'],
                            scheme_Qp_in=E['scheme']['Qp_in'], scheme_Qp_out=E['scheme']['Qp_out'],
                            scheme_T2=E['scheme']['T2'])
        res['measured_Qp'] = dict(
            inner_wall=rates['aniso'][P['patch_names']['a_lo']]['total'] / ang,
            interface_out_of_aniso=-ia['total'] / ang, interface_into_iso=ii['total'] / ang,
            outer_wall=-rates['iso'][P['patch_names']['i_hi']]['total'] / ang)

    # steady-state check for the pseudo-time-marched problems
    if P['problem'] in ('patch', 'annulus') and len(c.times) >= 2:
        prev = c.times[-2][1]
        d = max(float(np.max(np.abs(T[r] - c.field(prev, r, 'T')[0]))) for r in T)
        res['steady_change_since_previous_write'] = d

    # ---------------------------------------------------------- transient audit
    if P['problem'] == 'slab':
        hist = []
        for tv, tn in c.times:
            Ea = c.energy('aniso', c.field(tn, 'aniso', 'T')[0])
            Ei = c.energy('iso', c.field(tn, 'iso', 'T')[0])
            hist.append((tv, Ea, Ei))
        hist = np.array(hist)
        dEa, dEi = hist[-1, 1] - hist[0, 1], hist[-1, 2] - hist[0, 2]
        aud = dict(times=jf(hist[:, 0]), E_aniso=jf(hist[:, 1]), E_iso=jf(hist[:, 2]),
                   dE_aniso=float(dEa), dE_iso=float(dEi), drift=float(dEa + dEi),
                   ratio_received_over_released=float(-dEi / dEa) if dEa != 0 else None,
                   predicted_ratio_continuum=E['kc'] / E['knn'])
        # 1-D references on the solver's mesh and time step
        Hh = P['H']
        st = fv1d.Strip1D(E['knn'], E['kc'], E['Ca'], P['La'], P['nxa'], P['ki'], E['Ci'], P['Li'], P['nxi'],
                          P['xbc'][0], P['xbc'][1], P['TL'], P['TR'],
                          qr=(-E['knt'] * P['B'] if P['coupling'] == 'conormal' else 0.0))
        x1 = st.x()
        T0 = gencase.initial_T(P, E, x1, np.zeros_like(x1))
        order = tuple(P['order'].split())
        Tpart, eh = st.partitioned(T0, P['dt'], P['nsteps'], npass=P['nouter'], order=order, history=True)
        Tmono = st.monolithic(T0, P['dt'], P['nsteps'])
        Tsemi = st.semidiscrete(T0, tval)
        # solver field reduced to columns (subtract B*eta, average over eta)
        cols = []
        for r, n1 in (('aniso', P['nxa']), ('iso', P['nxi'])):
            xi, eta = c.coords(c.mesh[r].C)
            idx = np.argsort(xi, kind='stable')
            v = (T[r] - P['B'] * eta)[idx].reshape(n1, -1)
            cols.append(v.mean(axis=1))
            aud[f'spread_across_eta_{r}'] = float(np.max(np.ptp(v, axis=1)))
        Tof = np.concatenate(cols)
        ser_s = E['scheme_obj'].T(E['scheme_coef'], x1, tval) if E['scheme_coef'] is not None else None
        ser_e = E['obj'].T(E['coef'], x1, tval)
        mx = lambda a, b: float(np.max(np.abs(a - b)))
        split = dict(solver_minus_emulator=mx(Tof, Tpart),
                     staggering__emulator_minus_monolithic=mx(Tpart, Tmono),
                     euler__monolithic_minus_semidiscrete=mx(Tmono, Tsemi),
                     defect_plus_space__semidiscrete_minus_series_exact=mx(Tsemi, ser_e),
                     solver_minus_series_exact=mx(Tof, ser_e))
        if ser_s is not None:
            split['space__semidiscrete_minus_series_scheme'] = mx(Tsemi, ser_s)
            split['conservation_defect__series_scheme_minus_series_exact'] = mx(ser_s, ser_e)
            split['solver_minus_series_scheme'] = mx(Tof, ser_s)
        aud['split_Linf_K'] = split
        Ea0, Ei0 = st.energies(T0)
        Eap, Eip = st.energies(Tpart)
        aud['emulator'] = dict(dE_aniso=float((Eap - Ea0) * Hh), dE_iso=float((Eip - Ei0) * Hh),
                               drift=float((Eap + Eip - Ea0 - Ei0) * Hh),
                               ratio=float(-(Eip - Ei0) / (Eap - Ea0)) if Eap != Ea0 else None)
        if P['xbc'] == 'NN':
            g_ = E['kc'] / E['knn']
            aud['T_end_mean'] = float((hist[-1, 1] + hist[-1, 2]) / (E['Ca'] * P['La'] * Hh + E['Ci'] * P['Li'] * Hh))
            aud['Teq_conservative'] = ex.closed_pair_equilibrium(E['Ca'] * P['La'], E['Ci'] * P['Li'], P['Ta0'], P['Ti0'])
            aud['Teq_scheme'] = ex.closed_pair_equilibrium(E['Ca'] * P['La'], E['Ci'] * P['Li'], P['Ta0'], P['Ti0'], g_)
        res['audit'] = aud

    # ---------------------------------------------------------- solver log facts
    log = os.path.join(case, 'log.chtMultiRegionFoam')
    if os.path.exists(log):
        txt = open(log).read()
        res['log'] = dict(steps=txt.count('\nTime = '), ended='\nEnd' in txt,
                          h_solves=txt.count('Solving for h'),
                          wall_s=float(txt.rsplit('ClockTime = ', 1)[1].split()[0]) if 'ClockTime = ' in txt else None)
    json.dump(res, open(os.path.join(case, 'result.json'), 'w'), indent=1)
    return res


def summary(res):
    P = res['params']
    out = [f"case {os.path.basename(res['case'])}: {P['problem']} coupling={P['coupling']} nouter={P['nouter']} "
           f"t={res['time']:g} cells={res['cells']}"]
    for which, e in res['error'].items():
        out.append(f"  error vs {which:6s}: L2 {e['all']['L2']:.3e} K  Linf {e['all']['Linf']:.3e} K "
                   f"(scale {res['dT_scale']:g} K)")
    i = res['interface']
    if P['problem'] == 'slab':
        # transient: the stored patch temperature lags the cell values, so use the rates
        # recomputed from the two first-cell temperatures of the same instant
        out.append(f"  interface at t_end [W/m], recomputed from first-cell temperatures: into aniso "
                   f"{i['recomputed_into_aniso']:+.6f}; into iso {i['recomputed_into_iso']:+.6f}; "
                   f"imbalance {i['recomputed_imbalance']:+.3e}")
    else:
        out.append(f"  interface [W/m]: into aniso {i['into_aniso']:+.6f} (normal {i['into_aniso_normal']:+.6f}, "
                   f"tangential {i['into_aniso_tangential']:+.6f}); into iso {i['into_iso']:+.6f}; "
                   f"imbalance {i['imbalance']:+.3e}")
    if 'stock_wallHeatFlux_aniso' in i:
        out.append(f"  stock wallHeatFlux [W/m]: aniso {i['stock_wallHeatFlux_aniso']:+.6f}  iso {i.get('stock_wallHeatFlux_iso', float('nan')):+.6f}")
    if 'valueFraction_maxdiff' in i:
        out.append(f"  valueFraction(aniso): written {i['valueFraction_aniso_written']:.12f} predicted "
                   f"{i['valueFraction_aniso_predicted']:.12f}")
    if P['problem'] != 'slab':
        out.append(f"  steady audit per region, all patches [W/m]: aniso {res['region_audit']['aniso']:+.3e}  "
                   f"iso {res['region_audit']['iso']:+.3e}")
    if 'steady_change_since_previous_write' in res:
        out.append(f"  change since previous write: {res['steady_change_since_previous_write']:.3e} K")
    if 'audit' in res:
        a = res['audit']
        out.append(f"  energy [J/m]: dE_aniso {a['dE_aniso']:+.6f} dE_iso {a['dE_iso']:+.6f} drift {a['drift']:+.3e} "
                   f"ratio {a['ratio_received_over_released']} (continuum prediction {a['predicted_ratio_continuum']:.8f}, "
                   f"emulator {a['emulator']['ratio']})")
        for k, v in a['split_Linf_K'].items():
            out.append(f"    {k:55s} {v:.3e} K")
    imb = i['recomputed_imbalance'] if P['problem'] == 'slab' else i['imbalance']
    out.append('RESULT ' + ' '.join(f"Linf_vs_{k}={v['all']['Linf']:.3e}" for k, v in res['error'].items())
               + f" interface_imbalance={imb:+.3e}")
    return '\n'.join(out)


if __name__ == '__main__':
    print(summary(analyse(sys.argv[1])))
