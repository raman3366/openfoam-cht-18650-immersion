#!/usr/bin/env python3
"""
fv1d.py -- one-dimensional finite-volume references for the strip problems.

Same spatial discretisation as OpenFOAM on a uniform orthogonal mesh
(cell-centred, two-point fluxes, half-cell distance at walls), with the
interface treated exactly as the conductance-exchange coupling does
(DESIGN.md section 2):

    wall balance   b_a (Tw - Ta) + b_i (Tw - Ti) = qr        b = kappa_BC / d
    into aniso     c_a (Tw - Ta)                             c = Knn / d  (stencil)
    into iso       c_i (Tw - Ti)                             c_i = b_i = ki / d_i

Three time integrators over the same spatial operator:
    semidiscrete()  exact in time (matrix exponential)      -> spatial error only
    monolithic()    implicit Euler, both regions in one system -> + Euler error
    partitioned()   implicit Euler per region, neighbour cell temperature taken
                    from the other region as chtMultiRegionFoam does: the region
                    solved first sees the neighbour's previous value
                                                            -> + staggering error
Differences between consecutive levels isolate each error.  partitioned() is an
emulator of the solver's algorithm, not of its code; agreement with OpenFOAM is
measured by post.py, not assumed.

Per unit interface area; temperatures in K, energies in J/m2.
"""
import numpy as np


class Strip1D:
    def __init__(self, Knn, kc, Ca, La, na, ki, Ci, Li, ni,
                 left='N', right='N', TL=0.0, TR=0.0, qr=0.0):
        self.Knn, self.kc, self.Ca, self.La, self.na = float(Knn), float(kc), float(Ca), float(La), int(na)
        self.ki, self.Ci, self.Li, self.ni = float(ki), float(Ci), float(Li), int(ni)
        self.left, self.right, self.TL, self.TR, self.qr = left, right, float(TL), float(TR), float(qr)
        self.ha, self.hi = self.La / self.na, self.Li / self.ni
        self.xa = -self.La + (np.arange(self.na) + 0.5) * self.ha
        self.xi = (np.arange(self.ni) + 0.5) * self.hi
        da, di = self.ha / 2, self.hi / 2
        self.ca, self.ba = self.Knn / da, self.kc / da
        self.ci = self.bi = self.ki / di
        self.fa = self.bi / (self.ba + self.bi)      # valueFraction written by 'aniso'
        self.fi = self.ba / (self.ba + self.bi)      # valueFraction written by 'iso'
        self._assemble()

    def _assemble(self):
        na, ni = self.na, self.ni
        # region operators WITHOUT the interface exchange: C h dT/dt = L T + s
        La_ = np.zeros((na, na)); sa = np.zeros(na)
        g = self.Knn / self.ha
        for j in range(na - 1):
            La_[j, j] -= g; La_[j, j + 1] += g
            La_[j + 1, j + 1] -= g; La_[j + 1, j] += g
        if self.left == 'D':
            La_[0, 0] -= 2 * g; sa[0] += 2 * g * self.TL
        Li_ = np.zeros((ni, ni)); si = np.zeros(ni)
        g = self.ki / self.hi
        for j in range(ni - 1):
            Li_[j, j] -= g; Li_[j, j + 1] += g
            Li_[j + 1, j + 1] -= g; Li_[j + 1, j] += g
        if self.right == 'D':
            Li_[ni - 1, ni - 1] -= 2 * g; si[ni - 1] += 2 * g * self.TR
        self.La_, self.sa, self.Li_, self.si = La_, sa, Li_, si
        # interface exchange: into aniso = Ga (Ti0 - TaN) + ra ; into iso = Gi (TaN - Ti0) + ri
        self.Ga = self.ca * self.fa
        self.Gi = self.ci * self.fi
        self.ra = self.ca * self.qr / (self.ba + self.bi)
        self.ri = self.ci * self.qr / (self.ba + self.bi)
        # coupled system  dT/dt = M T + s
        n = na + ni
        A = np.zeros((n, n)); s = np.zeros(n)
        A[:na, :na] = La_; A[na:, na:] = Li_
        s[:na] = sa; s[na:] = si
        A[na - 1, na - 1] -= self.Ga; A[na - 1, na] += self.Ga; s[na - 1] += self.ra
        A[na, na] -= self.Gi; A[na, na - 1] += self.Gi; s[na] += self.ri
        cap = np.concatenate((np.full(na, self.Ca * self.ha), np.full(ni, self.Ci * self.hi)))
        self.cap = cap
        self.M = A / cap[:, None]
        self.s = s / cap

    # ---- helpers
    def x(self):
        return np.concatenate((self.xa, self.xi))

    def split(self, T):
        return T[:self.na], T[self.na:]

    def energies(self, T):
        Ta, Ti = self.split(T)
        return self.Ca * self.ha * Ta.sum(), self.Ci * self.hi * Ti.sum()

    def wall_temperature(self, T):
        Ta, Ti = self.split(T)
        return (self.ba * Ta[-1] + self.bi * Ti[0] + self.qr) / (self.ba + self.bi)

    # ---- integrators
    def steady(self):
        """Steady state of the coupled system (needs at least one fixed-temperature end)."""
        return np.linalg.solve(self.M, -self.s)

    def semidiscrete(self, T0, t):
        from scipy.linalg import expm
        n = len(T0)
        aug = np.zeros((n + 1, n + 1))
        aug[:n, :n] = self.M; aug[:n, n] = self.s
        return (expm(aug * t) @ np.concatenate((T0, [1.0])))[:n]

    def monolithic(self, T0, dt, nsteps):
        n = len(T0)
        B = np.linalg.inv(np.eye(n) - dt * self.M)
        T = T0.copy()
        for _ in range(nsteps):
            T = B @ (T + dt * self.s)
        return T

    def partitioned(self, T0, dt, nsteps, npass=1, order=('iso', 'aniso'), history=False):
        """chtMultiRegionFoam ordering: regions are solved one after another in the
        order given; each takes the neighbour's most recent first-cell temperature.
        npass = nOuterCorrectors.  Returns T (and per-step energies if history)."""
        na, ni = self.na, self.ni
        Ia, Ii = np.eye(na), np.eye(ni)
        capa, capi = self.Ca * self.ha, self.Ci * self.hi
        Aa = self.La_.copy(); Aa[na - 1, na - 1] -= self.Ga
        Ai = self.Li_.copy(); Ai[0, 0] -= self.Gi
        Ba = np.linalg.inv(capa / dt * Ia - Aa)
        Bi = np.linalg.inv(capi / dt * Ii - Ai)
        Ta, Ti = T0[:na].copy(), T0[na:].copy()
        hist = []
        for _ in range(nsteps):
            Ta_old, Ti_old = Ta.copy(), Ti.copy()
            for _p in range(npass):
                for reg in order:
                    if reg == 'iso':
                        rhs = capi / dt * Ti_old + self.si
                        rhs[0] += self.Gi * Ta[-1] + self.ri
                        Ti = Bi @ rhs
                    else:
                        rhs = capa / dt * Ta_old + self.sa
                        rhs[na - 1] += self.Ga * Ti[0] + self.ra
                        Ta = Ba @ rhs
            if history:
                hist.append((capa * Ta.sum(), capi * Ti.sum()))
        T = np.concatenate((Ta, Ti))
        return (T, np.array(hist)) if history else T


# =========================================================================== checks


def selfcheck():
    import math
    from exact import TwoLayer, twolayer_for_scheme, knn_knt, frob
    res = []

    def ok(name, val, tol):
        p = bool(abs(val) <= tol)
        res.append(p)
        print(f"  [{'ok' if p else 'FAIL'}] {name:66s} {val: .3e}  (tol {tol:g})")

    k, th = (28.0, 0.8, 0.8), 90.0
    Knn, _ = knn_knt(k, th)
    Ca, Ci, La, Li, ki = 2.0e6, 1.5e6, 0.004, 0.006, 0.15
    Ta0, Ti0 = 340.0, 305.0
    tau = (La * math.sqrt(Ca / Knn) + Li * math.sqrt(Ci / ki)) ** 2
    t = 0.05 * tau

    print('series against the semi-discrete finite-volume solution (spatial order)')
    for kc, tag in ((Knn, 'conservative'), (frob(k), 'scalar option')):
        tl = twolayer_for_scheme(k, th, Ca, La, ki, Ci, Li, kc=kc, left='N', right='N', nmodes=600)
        coef = tl.coefficients((Ta0, 0.0), (Ti0, 0.0))
        errs = []
        for n in (20, 40, 80):
            s = Strip1D(Knn, kc, Ca, La, 2 * n, ki, Ci, Li, 3 * n)
            T0 = np.concatenate((np.full(s.na, Ta0), np.full(s.ni, Ti0)))
            T = s.semidiscrete(T0, t)
            errs.append(np.max(np.abs(T - tl.T(coef, s.x(), t))))
        p1, p2 = math.log2(errs[0] / errs[1]), math.log2(errs[1] / errs[2])
        print(f'    {tag}: Linf error {errs[0]:.3e} {errs[1]:.3e} {errs[2]:.3e}, orders {p1:.2f} {p2:.2f}')
        ok(f'{tag}: second order in space (|p-2| < 0.25)', max(abs(p2 - 2) - 0.25, 0), 0)
        ok(f'{tag}: finest-mesh error small', errs[2], 5e-3)

    print('Dirichlet step with interface source (sheared strip) against the series')
    q = 123.0
    tl = TwoLayer(Knn, Ca, La, ki, Ci, Li, 'D', 'D', TL=350.0, TR=300.0, qjump=q, nmodes=600)
    coef = tl.coefficients((320.0, 0.0), (320.0, 0.0))
    errs = []
    for n in (20, 40, 80):
        s = Strip1D(Knn, Knn, Ca, La, 2 * n, ki, Ci, Li, 3 * n, 'D', 'D', 350.0, 300.0, qr=-q)
        T0 = np.full(s.na + s.ni, 320.0)
        errs.append(np.max(np.abs(s.semidiscrete(T0, t) - tl.T(coef, s.x(), t))))
    p2 = math.log2(errs[1] / errs[2])
    print(f'    Linf error {errs[0]:.3e} {errs[1]:.3e} {errs[2]:.3e}, order {p2:.2f}')
    ok('interface source: second order (|p-2| < 0.25)', max(abs(p2 - 2) - 0.25, 0), 0)
    ok('interface source: steady wall temperature', s.wall_temperature(s.steady()) - tl.Tw_ss, 2e-9)

    print('time integrators')
    s = Strip1D(Knn, Knn, Ca, La, 16, ki, Ci, Li, 24)
    T0 = np.concatenate((np.full(s.na, Ta0), np.full(s.ni, Ti0)))
    Tsd = s.semidiscrete(T0, t)
    e = [np.max(np.abs(s.monolithic(T0, t / n, n) - Tsd)) for n in (50, 100, 200)]
    ok('implicit Euler first order (|p-1| < 0.1)', max(abs(math.log2(e[1] / e[2]) - 1) - 0.1, 0), 0)
    dt, n = t / 100, 100
    Tm = s.monolithic(T0, dt, n)
    d = [np.max(np.abs(s.partitioned(T0, dt, n, npass=m) - Tm)) for m in (1, 2, 3, 6)]
    print(f'    partitioned - monolithic, 1/2/3/6 passes: {d[0]:.3e} {d[1]:.3e} {d[2]:.3e} {d[3]:.3e}')
    rho_c = d[1] / d[0]
    ok('coupling passes contract geometrically (6 passes vs rho^5)', max(d[3] - 3 * d[0] * rho_c ** 5, 0), 0)
    ok('contraction factor < 1', max(rho_c - 0.999, 0), 0)
    ok('one pass differs from monolithic (staggering error present)', 0.0 if d[0] > 1e-6 else 1.0, 0)
    Ea0, Ei0 = s.energies(T0)
    for m in (1, 2, 3):
        Tp = s.partitioned(T0, dt, n, npass=m)
        Ea, Ei = s.energies(Tp)
        print(f'    passes={m}: energy drift (Ea+Ei - initial) = {Ea + Ei - Ea0 - Ei0:+.6e} J/m2')
    # staggering identity (iso solved first, aniso second, conservative coupling):
    #   dE_i = dt G (Ta^n - Ti^{n+1}),  dE_a = dt G (Ti^{n+1} - Ta^{n+1})
    #   => sum telescopes to  -dt G (Ta_N(end) - Ta_N(0)),  Ta_N = aniso first cell
    Tp = s.partitioned(T0, dt, n, npass=1)
    Ea, Ei = s.energies(Tp)
    pred = -dt * s.Gi * (Tp[s.na - 1] - T0[s.na - 1])
    ok('one-pass energy drift = -dt*G*(Ta_N(end) - Ta_N(0))  [iso solved first]',
       (Ea + Ei - Ea0 - Ei0 - pred) / abs(pred), 1e-9)
    print('scalar option: energy ratio')
    s = Strip1D(Knn, frob(k), Ca, La, 16, ki, Ci, Li, 24)
    Tp = s.semidiscrete(T0, t)
    Ea, Ei = s.energies(Tp)
    ok('semi-discrete: -dEi/dEa = |K|/Knn', -(Ei - Ei0) / (Ea - Ea0) - frob(k) / Knn, 1e-8)
    nfail = res.count(False)
    print(f'\n{len(res)} checks, {nfail} failed')
    return nfail


if __name__ == '__main__':
    import sys
    sys.exit(1 if selfcheck() else 0)
