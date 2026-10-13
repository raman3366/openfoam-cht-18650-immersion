#!/usr/bin/env python3
"""
exact.py -- exact solutions for the two-region conduction verification problems.
Pure numpy/scipy, no OpenFOAM needed.

Geometry conventions used by gencase.py / post.py
-------------------------------------------------
strip problems (P1 patch test, P3 transient, P4 flux-ratio):
    region 'aniso' : xi in [-La, 0]   conductivity tensor K = R(theta) diag(k1,k2,k3) R(theta)^T
    region 'iso'   : xi in [0,  Li]   scalar conductivity ki
    interface      : xi = 0, unit normal n = +e_xi (out of 'aniso'), tangent e_eta
    theta          : angle (deg) of principal axis 1 from the interface normal, rotation about z
    Knn = n.K.n = k1 cos^2 + k2 sin^2,  Knt = (k1 - k2) sin cos,  Ktt = k1 sin^2 + k2 cos^2
annulus (P2):
    region 'aniso' : r in [r1, r2]    cylindrical orthotropy (kr, kth, kz)
    region 'iso'   : r in [r2, r3]    scalar ko

Every "scheme" solution below is the exact solution of the transmission problem that
a coupling with a *coupling conductivity* kc (instead of Knn) and with or without the
tangential co-normal term enforces.  See DESIGN.md section 3 for the derivations.

Run `python3 exact.py` for the unit checks.
"""
import math
import numpy as np

# --------------------------------------------------------------------------- tensors


def rotz(theta_deg):
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def K_tensor(k, theta_deg):
    """3x3 conductivity tensor, principal values k=(k1,k2,k3), axis 1 at theta from +x."""
    R = rotz(theta_deg)
    return R @ np.diag(np.asarray(k, float)) @ R.T


def knn_knt(k, theta_deg):
    """Normal and normal-tangential components for the interface normal +x."""
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    return k[0] * c * c + k[1] * s * s, (k[0] - k[1]) * s * c


def frob(k):
    """What constAnIso returns as scalar kappa: mag of the principal vector."""
    return math.sqrt(sum(float(v) ** 2 for v in k))


def coupling_kappa(method, k, theta_deg, kc_value=None):
    """Conductivity the coupled condition is given by each kappaMethod, normal = +x."""
    if method == 'solidThermo':
        return frob(k)
    if method in ('directionalSolidThermo', 'conormal'):
        return knn_knt(k, theta_deg)[0]
    if method == 'function':
        return float(kc_value)
    raise ValueError(method)


# --------------------------------------------------------------------------- P1


class StripSteady:
    """Steady strip, T = phi(xi) + B*eta, phi piecewise linear.

    Outer data: T(-La, eta) = TL + B eta,  T(Li, eta) = TR + B eta.
    Exact transmission conditions at xi = 0:
        [T] = 0,   ki dT/dxi|iso = Knn dT/dxi|aniso + Knt B.
    """

    def __init__(self, k, theta_deg, ki, La, Li, TL, TR, B):
        self.k, self.theta, self.ki = tuple(float(v) for v in k), float(theta_deg), float(ki)
        self.La, self.Li, self.TL, self.TR, self.B = map(float, (La, Li, TL, TR, B))
        self.Knn, self.Knt = knn_knt(self.k, self.theta)

    @classmethod
    def from_gradient(cls, k, theta_deg, ki, La, Li, T0, A, B):
        """Patch-test form: T0 at the interface (eta=0), normal gradient A in 'aniso'."""
        Knn, Knt = knn_knt(k, theta_deg)
        Ai = (Knn * A + Knt * B) / ki
        return cls(k, theta_deg, ki, La, Li, T0 - A * La, T0 + Ai * Li, B)

    def _solve(self, kc, qjump):
        # kc*Aa - ki*Ai = -qjump ;  Aa = (Tw-TL)/La ; Ai = (TR-Tw)/Li
        ga, gi = kc / self.La, self.ki / self.Li
        Tw = (ga * self.TL + gi * self.TR - qjump) / (ga + gi)
        Aa, Ai = (Tw - self.TL) / self.La, (self.TR - Tw) / self.Li
        return dict(Tw=Tw, Aa=Aa, Ai=Ai,
                    # heat flux densities in +xi [W/m2]
                    q_aniso_normal=-self.Knn * Aa,
                    q_aniso_tangential=-self.Knt * self.B,
                    q_aniso=-(self.Knn * Aa + self.Knt * self.B),
                    q_iso=-self.ki * Ai)

    def exact(self):
        return self._solve(self.Knn, self.Knt * self.B)

    def scheme(self, kc, conormal=False):
        """What a conductance-exchange coupling with conductivity kc converges to
        when the y-walls carry the exact gradient B (sheared strip).  conormal=True
        adds the tangential term to the wall balance (requires kc = Knn to be exact)."""
        return self._solve(kc, self.Knt * self.B if conormal else 0.0)

    def T(self, sol, xi, eta):
        xi = np.asarray(xi, float)
        return np.where(xi <= 0.0, sol['Tw'] + sol['Aa'] * xi, sol['Tw'] + sol['Ai'] * xi) \
            + self.B * np.asarray(eta, float)


# --------------------------------------------------------------------------- P2


class Annulus:
    """Steady radial conduction, T(r1)=T1, T(r3)=T3, interface at r2.

    Exact:  T_in  = T1 - Qp ln(r/r1)/kr,  T_out = T3 + Qp_out ln(r3/r)/ko,
            Qp = -kr r dT/dr  [W per metre depth per radian], constant in each layer.
    Scheme with coupling conductivity kc: kc dT/dr|in = ko dT/dr|out, hence
            Qp_out = (kc/kr) Qp_in.
    """

    def __init__(self, r1, r2, r3, kr, ko, T1, T3):
        self.r1, self.r2, self.r3, self.kr, self.ko, self.T1, self.T3 = map(
            float, (r1, r2, r3, kr, ko, T1, T3))

    def solve(self, kc=None):
        g = 1.0 if kc is None else kc / self.kr
        Ra = math.log(self.r2 / self.r1) / self.kr
        Ro = math.log(self.r3 / self.r2) / self.ko
        Qin = (self.T1 - self.T3) / (Ra + g * Ro)
        return dict(Qp_in=Qin, Qp_out=g * Qin, T2=self.T1 - Qin * Ra, gamma=g)

    def T(self, sol, r):
        r = np.asarray(r, float)
        Tin = self.T1 - sol['Qp_in'] * np.log(r / self.r1) / self.kr
        Tout = self.T3 + sol['Qp_out'] * np.log(self.r3 / r) / self.ko
        return np.where(r <= self.r2, Tin, Tout)


# --------------------------------------------------------------------------- P3


def _phi(kind, z):
    return np.sin(z) if kind == 'D' else np.cos(z)


def _dphi(kind, z):
    return np.cos(z) if kind == 'D' else -np.sin(z)


class TwoLayer:
    """Transient conduction in a two-layer slab by eigenfunction expansion.

    layer 1: x in [-L1, 0], conductivity k1, volumetric heat capacity C1 = rho*cp
    layer 2: x in [0, L2],  conductivity k2, C2
    outer conditions 'D' (fixed temperature) or 'N' (zero normal flux) at each end
    interface: [T] = 0,  k2 T_x(0+) - k1 T_x(0-) = qjump   (qjump = Knt*B for the
    sheared strip, 0 otherwise)

    T(x,t) = Tp(x) + sum_n c_n X_n(x) exp(-w_n^2 t)
    X_n = a_n phi1(b1 s1) in layer 1 (s1 = x + L1), b_n phi2(b2 s2) in layer 2 (s2 = L2 - x),
    b_i = w/sqrt(alpha_i).  Orthogonality weight is C_i.
    """

    def __init__(self, k1, C1, L1, k2, C2, L2, left='N', right='N',
                 TL=0.0, TR=0.0, qjump=0.0, nmodes=400):
        self.k1, self.C1, self.L1, self.k2, self.C2, self.L2 = map(float, (k1, C1, L1, k2, C2, L2))
        self.left, self.right = left, right
        self.TL, self.TR, self.qjump = float(TL), float(TR), float(qjump)
        self.a1, self.a2 = self.k1 / self.C1, self.k2 / self.C2
        self.e1, self.e2 = math.sqrt(self.k1 * self.C1), math.sqrt(self.k2 * self.C2)
        self.t1, self.t2 = self.L1 / math.sqrt(self.a1), self.L2 / math.sqrt(self.a2)
        if left == 'N' and right == 'N' and qjump != 0.0:
            raise ValueError('no steady state: closed slab with an interface source')
        self._roots(nmodes)
        self._particular()

    # characteristic function F(w) = e1 phi1'(w t1) phi2(w t2) + e2 phi2'(w t2) phi1(w t1)
    def F(self, w):
        return (self.e1 * _dphi(self.left, w * self.t1) * _phi(self.right, w * self.t2)
                + self.e2 * _dphi(self.right, w * self.t2) * _phi(self.left, w * self.t1))

    def _roots(self, nmodes):
        from scipy.optimize import brentq
        tsum = self.t1 + self.t2
        dw = math.pi / tsum / 400.0           # 400 samples per mean root spacing
        roots, w0 = [], dw * 1e-6
        f0 = self.F(w0)
        chunk = 200000
        wmax_guess = (nmodes + 2) * math.pi / tsum
        while len(roots) < nmodes:
            w = w0 + dw * np.arange(1, chunk + 1)
            f = self.F(w)
            fa = np.concatenate(([f0], f[:-1]))
            wa = np.concatenate(([w0], w[:-1]))
            idx = np.nonzero(np.sign(fa) * np.sign(f) < 0)[0]
            for i in idx:
                roots.append(brentq(self.F, wa[i], w[i], xtol=1e-15 * max(1.0, w[i]), rtol=1e-15))
                if len(roots) >= nmodes:
                    break
            w0, f0 = w[-1], f[-1]
            if w0 > 50 * wmax_guess:
                break
        self.w = np.array(roots[:nmodes])
        self.zero_mode = (self.left == 'N' and self.right == 'N')
        # amplitudes (a, b): null vector of [[phi1, -phi2], [e1 phi1', e2 phi2']]
        z1, z2 = self.w * self.t1, self.w * self.t2
        p1, p2 = _phi(self.left, z1), _phi(self.right, z2)
        d1, d2 = self.e1 * _dphi(self.left, z1), self.e2 * _dphi(self.right, z2)
        use1 = (p1 ** 2 + p2 ** 2) >= (d1 ** 2 + d2 ** 2) / max(self.e1, self.e2) ** 2
        a = np.where(use1, p2, d2)
        b = np.where(use1, p1, -d1)
        nrm = np.sqrt(a * a + b * b)
        self.a, self.b = a / nrm, b / nrm
        self.b1, self.b2 = self.w / math.sqrt(self.a1), self.w / math.sqrt(self.a2)
        # norms  N = C1 a^2 int phi1^2 + C2 b^2 int phi2^2
        sg1 = -1.0 if self.left == 'D' else 1.0
        sg2 = -1.0 if self.right == 'D' else 1.0
        I1 = self.L1 / 2 + sg1 * np.sin(2 * self.b1 * self.L1) / (4 * self.b1)
        I2 = self.L2 / 2 + sg2 * np.sin(2 * self.b2 * self.L2) / (4 * self.b2)
        self.N = self.C1 * self.a ** 2 * I1 + self.C2 * self.b ** 2 * I2

    def _particular(self):
        k1, k2, L1, L2, q = self.k1, self.k2, self.L1, self.L2, self.qjump
        if self.left == 'D' and self.right == 'D':
            Tw = (k1 * self.TL / L1 + k2 * self.TR / L2 - q) / (k1 / L1 + k2 / L2)
            A1, A2 = (Tw - self.TL) / L1, (self.TR - Tw) / L2
        elif self.left == 'D':                      # right insulated
            A2, A1 = 0.0, -q / k1
            Tw = self.TL + A1 * L1
        elif self.right == 'D':                     # left insulated
            A1, A2 = 0.0, q / k2
            Tw = self.TR - A2 * L2
        else:                                       # closed: fixed later from energy
            A1 = A2 = 0.0
            Tw = 0.0
        self.Tw_ss, self.A1_ss, self.A2_ss = Tw, A1, A2

    def Tp(self, x):
        x = np.asarray(x, float)
        return np.where(x <= 0, self.Tw_ss + self.A1_ss * x, self.Tw_ss + self.A2_ss * x)

    def X(self, n, x):
        x = np.asarray(x, float)
        return np.where(x <= 0,
                        self.a[n] * _phi(self.left, self.b1[n] * (x + self.L1)),
                        self.b[n] * _phi(self.right, self.b2[n] * (self.L2 - x)))

    @staticmethod
    def _int_lin(kind, beta, L, p, q):
        """int_0^L (p + q s) phi(beta s) ds, phi = sin (D) or cos (N)."""
        if kind == 'N':
            return p * np.sin(beta * L) / beta + q * (L * np.sin(beta * L) / beta
                                                      + (np.cos(beta * L) - 1) / beta ** 2)
        return p * (1 - np.cos(beta * L)) / beta + q * (-L * np.cos(beta * L) / beta
                                                        + np.sin(beta * L) / beta ** 2)

    def coefficients(self, ic1, ic2):
        """Initial condition, piecewise linear: layer 1  T = ic1[0] + ic1[1]*x,
        layer 2  T = ic2[0] + ic2[1]*x  (x is the global coordinate, interface at 0)."""
        # deviation from the particular solution, as p + q*s in local wall coordinates
        v1, g1 = ic1[0] - self.Tw_ss, ic1[1] - self.A1_ss          # in x
        v2, g2 = ic2[0] - self.Tw_ss, ic2[1] - self.A2_ss
        # x = s1 - L1  ->  v1 + g1*(s1 - L1) ;   x = L2 - s2 -> v2 + g2*(L2 - s2)
        J1 = self._int_lin(self.left, self.b1, self.L1, v1 - g1 * self.L1, g1)
        J2 = self._int_lin(self.right, self.b2, self.L2, v2 + g2 * self.L2, -g2)
        c = (self.C1 * self.a * J1 + self.C2 * self.b * J2) / self.N
        c0 = 0.0
        if self.zero_mode:
            M1, M2 = self.C1 * self.L1, self.C2 * self.L2
            m1 = v1 - g1 * self.L1 / 2          # layer means of the deviation
            m2 = v2 + g2 * self.L2 / 2
            c0 = (M1 * m1 + M2 * m2) / (M1 + M2)
        return c, c0

    def T(self, coef, x, t):
        c, c0 = coef
        x = np.asarray(x, float)
        out = self.Tp(x) + c0
        dec = np.exp(-self.w ** 2 * t)
        keep = np.nonzero(np.abs(c) * dec > 0)[0]
        for n in keep:
            out = out + c[n] * dec[n] * self.X(n, x)
        return out

    def layer_means(self, coef, t):
        """Mean temperature of each layer (for the energy audit)."""
        c, c0 = coef
        dec = c * np.exp(-self.w ** 2 * t)
        m1 = self.Tw_ss - self.A1_ss * self.L1 / 2 + c0 + np.sum(
            dec * self.a * self._int_lin(self.left, self.b1, self.L1, 1.0, 0.0)) / self.L1
        m2 = self.Tw_ss + self.A2_ss * self.L2 / 2 + c0 + np.sum(
            dec * self.b * self._int_lin(self.right, self.b2, self.L2, 1.0, 0.0)) / self.L2
        return m1, m2

    def tail_bound(self, coef, t):
        """Size of the last retained term (series truncation indicator)."""
        c, _ = coef
        return float(abs(c[-1]) * math.exp(-self.w[-1] ** 2 * t))


def twolayer_for_scheme(k, theta_deg, Ca, La, ki, Ci, Li, kc=None, **kw):
    """TwoLayer for the strip.  kc=None: exact problem (layer 1 uses Knn, Ca).
    kc given: the transmission problem a conductance-exchange coupling with coupling
    conductivity kc enforces, kc T_x|aniso = ki T_x|iso with interior diffusivity
    Knn/Ca.  It is the conservative two-layer problem with layer 1 = (g*Knn, g*Ca),
    g = kc/Knn (multiply the layer-1 equation by g)."""
    Knn, _ = knn_knt(k, theta_deg)
    g = 1.0 if kc is None else kc / Knn
    return TwoLayer(g * Knn, g * Ca, La, ki, Ci, Li, **kw)


# --------------------------------------------------------------------------- P4


def flux_ratio(kc, k, theta_deg):
    """Heat received by the isotropic neighbour / heat released by the anisotropic
    solid through an aligned or 1-D interface: kc / (n.K.n)."""
    return kc / knn_knt(k, theta_deg)[0]


def closed_pair_equilibrium(Ma, Mi, Ta0, Ti0, g=1.0):
    """Final uniform temperature of an adiabatic pair (heat capacities per unit area
    Ma, Mi) when the neighbour receives g times what the solid releases (continuum
    limit, no staggering term)."""
    return (g * Ma * Ta0 + Mi * Ti0) / (g * Ma + Mi)


# --------------------------------------------------------------------------- extension


class AnisoInclusion:
    """Circular inclusion (radius a) with K = diag(k1,k2) in fixed Cartesian principal
    axes, embedded in an isotropic medium km under a uniform far-field gradient G.
    Interior gradient is uniform:  g_j = 2 km G_j /(km + k_j);  exterior:
    T = G.x + (a^2/r^2) p.x,  p_j = G_j (km - k_j)/(km + k_j).   (x in principal axes.)
    Curved interface whose normal sweeps all angles to the principal axes: a non-degenerate
    test of the tangential term on a curved interface.  Not generated or run in this job."""

    def __init__(self, a, k1, k2, km, G1, G2, T0=0.0):
        self.a, self.k = float(a), (float(k1), float(k2))
        self.km, self.G, self.T0 = float(km), (float(G1), float(G2)), float(T0)
        self.g = tuple(2 * km * Gj / (km + kj) for Gj, kj in zip(self.G, self.k))
        self.p = tuple(Gj * (km - kj) / (km + kj) for Gj, kj in zip(self.G, self.k))

    def T(self, x1, x2):
        x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
        r2 = x1 * x1 + x2 * x2
        Tin = self.g[0] * x1 + self.g[1] * x2
        with np.errstate(divide='ignore', invalid='ignore'):
            Tout = self.G[0] * x1 + self.G[1] * x2 + self.a ** 2 * (self.p[0] * x1 + self.p[1] * x2) / r2
        return self.T0 + np.where(r2 <= self.a ** 2, Tin, Tout)


# =========================================================================== checks


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def selfcheck(verbose=True):
    res = []

    def ok(name, val, tol):
        passed = bool(abs(val) <= tol)
        res.append((name, float(val), tol, passed))
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {name:62s} {val: .3e}  (tol {tol:g})")

    print('tensor algebra')
    k = (28.0, 0.8, 0.8)
    for th in (0, 30, 45, 60, 90):
        K = K_tensor(k, th)
        Knn, Knt = knn_knt(k, th)
        ok(f'Knn = K[0,0] at {th} deg', K[0, 0] - Knn, 1e-13)
        ok(f'Knt = K[0,1] at {th} deg', K[0, 1] - Knt, 1e-13)
        ok(f'frobenius invariant at {th} deg', math.sqrt(np.sum(np.linalg.eigvalsh(K) ** 2)) - frob(k), 1e-12)
    ok('|(0.8,0.8,28)| = sqrt(785.28) = 28.02285', frob(k) - math.sqrt(785.28), 1e-13)
    ok('... and equals 28.02285 to the printed digits', frob(k) - 28.02285, 5e-6)
    ok('ratio at 90 deg = 35.02856', flux_ratio(frob(k), k, 90) - 35.02856, 5e-6)
    ok('ratio at  0 deg = 1.000816', flux_ratio(frob(k), k, 0) - 1.000816, 5e-7)
    ok('(k,k,k) gives sqrt(3)', flux_ratio(frob((2.0, 2.0, 2.0)), (2.0, 2.0, 2.0), 37) - math.sqrt(3), 1e-14)

    print('P1 strip, steady')
    for th in (0, 30, 45, 60, 90):
        s = StripSteady.from_gradient(k, th, 1.0, 0.01, 0.01, 320.0, 1000.0, 500.0)
        e = s.exact()
        ok(f'theta={th}: interface temperature T0', e['Tw'] - 320.0, 1e-10)
        ok(f'theta={th}: total co-normal flux continuous', (e['q_aniso'] - e['q_iso']) / abs(e['q_iso']), 1e-13)
        # field: continuity and PDE (linear => trivially harmonic); check via FD of T
        f = lambda x: float(s.T(e, x, 0.003))
        ja = -(s.Knn * _fd(f, -0.004, 1e-5) + s.Knt * s.B)
        ji = -s.ki * _fd(f, 0.004, 1e-5)
        ok(f'theta={th}: FD flux balance', (ja - ji) / abs(ji), 1e-8)
        st = s.scheme(s.Knn)                           # stock directional, sheared strip
        ok(f'theta={th}: stock-directional wall error = Knt B/(Knn/La+ki/Li)',
           (st['Tw'] - e['Tw']) - s.Knt * s.B / (s.Knn / s.La + s.ki / s.Li), 1e-9)
        ok(f'theta={th}: stock creation rate = Knt B', (st['q_iso'] - st['q_aniso']) - s.Knt * s.B, 1e-9)
        co = s.scheme(s.Knn, conormal=True)
        ok(f'theta={th}: co-normal correction is exact', co['Tw'] - e['Tw'], 1e-10)
        sc = s.scheme(frob(k))
        ok(f'theta={th}: scalar option normal-flux ratio = |K|/Knn',
           sc['q_iso'] / sc['q_aniso_normal'] - frob(k) / s.Knn, 1e-9)

    print('P2 annulus')
    an = Annulus(0.004, 0.009, 0.015, 0.8, 0.15, 350.0, 300.0)
    e = an.solve()
    f = lambda r: float(an.T(e, r))
    ok('T(r1) = T1', f(0.004) - 350.0, 1e-12)
    ok('T(r3) = T3', f(0.015) - 300.0, 1e-12)
    ok('continuity at r2', f(0.009 * (1 - 1e-14)) - f(0.009 * (1 + 1e-14)), 1e-9)
    qi = -an.kr * 0.0089 * _fd(f, 0.0089, 1e-6)
    qo = -an.ko * 0.0091 * _fd(f, 0.0091, 1e-6)
    ok('r q_r equal on both sides (FD)', (qi - qo) / qo, 1e-7)
    for r0 in (0.006, 0.012):
        h = 1e-4                                   # (r T')' = 0
        lap = ((r0 + h / 2) * (f(r0 + h) - f(r0)) - (r0 - h / 2) * (f(r0) - f(r0 - h))) / h ** 2
        ok(f'radial Laplace residual at r={r0}', lap * 1e-6, 1e-3)
    d = an.solve(kc=frob((0.8, 0.8, 28.0)))
    ok('scheme: Qp_out/Qp_in = kc/kr', d['Qp_out'] / d['Qp_in'] - frob((0.8, 0.8, 28.0)) / 0.8, 1e-11)
    fd_ = lambda r: float(an.T(d, r))
    ok('scheme: continuity at r2', fd_(0.009 * (1 - 1e-14)) - fd_(0.009 * (1 + 1e-14)), 1e-9)

    print('P3 two-layer series')
    for (left, right) in (('N', 'N'), ('D', 'D'), ('D', 'N'), ('N', 'D')):
        for (k1, C1, k2, C2) in ((0.8, 2.0e6, 0.15, 1.5e6), (28.0, 2.0e6, 0.15, 1.5e6),
                                 (0.8 * 35.0, 2.0e6 * 35.0, 0.15, 1.5e6), (1000.0, 2.0e6, 1.0, 2.0e6)):
            tag = f'{left}{right} k1={k1:g} C1={C1:.1e}'
            q = 0.0 if (left, right) == ('N', 'N') else 37.0
            tl = TwoLayer(k1, C1, 0.004, k2, C2, 0.006, left, right, TL=350.0, TR=300.0, qjump=q, nmodes=300)
            # 1. eigen-equation residual and root count against Weyl's law
            ok(f'{tag}: max |F(w_n)| / scale', np.max(np.abs(tl.F(tl.w))) / (tl.e1 + tl.e2), 1e-9)
            weyl = tl.w[-1] * (tl.t1 + tl.t2) / math.pi
            ok(f'{tag}: root count vs Weyl (|diff| <= 1.5)', max(abs(len(tl.w) - weyl) - 1.5, 0.0), 0.0)
            # 2. interface conditions of each eigenfunction
            n = np.arange(len(tl.w))
            jump = tl.a * _phi(left, tl.b1 * tl.L1) - tl.b * _phi(right, tl.b2 * tl.L2)
            flx = tl.k1 * tl.a * tl.b1 * _dphi(left, tl.b1 * tl.L1) + tl.k2 * tl.b * tl.b2 * _dphi(right, tl.b2 * tl.L2)
            ok(f'{tag}: eigenfunction continuity', np.max(np.abs(jump)), 1e-8)
            ok(f'{tag}: eigenfunction flux continuity (rel)',
               np.max(np.abs(flx) / (tl.w * (tl.e1 + tl.e2))), 1e-8)
            # 3. orthogonality by quadrature for a few pairs
            xs1 = -tl.L1 + (np.arange(20000) + 0.5) * tl.L1 / 20000
            xs2 = (np.arange(20000) + 0.5) * tl.L2 / 20000
            def ip(m, n_):
                return (tl.C1 * np.sum(tl.X(m, xs1) * tl.X(n_, xs1)) * tl.L1 / 20000
                        + tl.C2 * np.sum(tl.X(m, xs2) * tl.X(n_, xs2)) * tl.L2 / 20000)
            off = max(abs(ip(0, 1)), abs(ip(1, 4)), abs(ip(2, 7))) / math.sqrt(tl.N[0] * tl.N[1])
            ok(f'{tag}: orthogonality (quadrature)', off, 2e-6)
            ok(f'{tag}: closed-form norm vs quadrature', (ip(3, 3) - tl.N[3]) / tl.N[3], 2e-6)
            # 4. initial condition reproduced (L2, away from the jump) and steady limit
            coef = tl.coefficients((340.0, 0.0), (305.0, 0.0))
            x = np.concatenate((np.linspace(-tl.L1 * 0.9, -tl.L1 * 0.1, 7), np.linspace(tl.L2 * 0.1, tl.L2 * 0.9, 7)))
            # (only meaningful when 300 modes resolve both layers at the test time;
            #  the stiff cases are cross-checked against the finite-volume reference
            #  in fv1d.py instead)
            if max(tl.t1, tl.t2) / min(tl.t1, tl.t2) < 5.0:
                tsmall = 1e-4 * min(tl.t1, tl.t2) ** 2
                T0 = tl.T(coef, x, tsmall)
                ok(f'{tag}: initial condition at t->0+ (interior pts)',
                   np.max(np.abs(T0 - np.where(x < 0, 340.0, 305.0))), 2e-2)
            # 5. PDE residual at a later time by finite differences
            tt = 0.05 * (tl.t1 + tl.t2) ** 2
            for (xc, al) in ((-0.5 * tl.L1, tl.a1), (0.5 * tl.L2, tl.a2)):
                hx, ht = 2e-5, tt * 1e-4
                Txx = (tl.T(coef, xc + hx, tt) - 2 * tl.T(coef, xc, tt) + tl.T(coef, xc - hx, tt)) / hx ** 2
                Tt = (tl.T(coef, xc, tt + ht) - tl.T(coef, xc, tt - ht)) / (2 * ht)
                ok(f'{tag}: PDE residual (rel) x={xc:+.4f}',
                   float(abs(Tt - al * Txx) / (abs(Tt) + 1e-30)), 5e-4)
            # 6. interface jump condition of the full solution
            #    (term-by-term derivative of the series; a finite difference is
            #     dominated by its own truncation error when C1 is large)
            dec = coef[0] * np.exp(-tl.w ** 2 * tt)
            d1 = tl.A1_ss + np.sum(dec * tl.a * tl.b1 * _dphi(left, tl.b1 * tl.L1))
            d2 = tl.A2_ss - np.sum(dec * tl.b * tl.b2 * _dphi(right, tl.b2 * tl.L2))
            ok(f'{tag}: k2 T_x(0+) - k1 T_x(0-) = qjump (series derivative)',
               float(tl.k2 * d2 - tl.k1 * d1 - q) / (abs(q) + tl.k2 * 35 / tl.L2), 1e-9)
            ok(f'{tag}: temperature continuous at the interface',
               float(tl.T(coef, -1e-12, tt) - tl.T(coef, 1e-12, tt)), 1e-6)
            # 7. energy: closed slab conserves C-weighted mean
            if (left, right) == ('N', 'N'):
                m1, m2 = tl.layer_means(coef, tt)
                E = tl.C1 * tl.L1 * m1 + tl.C2 * tl.L2 * m2
                E0 = tl.C1 * tl.L1 * 340.0 + tl.C2 * tl.L2 * 305.0
                ok(f'{tag}: closed-slab energy conserved (rel)', (E - E0) / E0, 1e-11)
                Teq = closed_pair_equilibrium(tl.C1 * tl.L1, tl.C2 * tl.L2, 340.0, 305.0)
                ok(f'{tag}: t->inf gives T_eq', float(tl.T(coef, 0.001, 1e3 * (tl.t1 + tl.t2) ** 2)) - Teq, 1e-9)
            else:
                ok(f'{tag}: t->inf gives steady solution',
                   float(tl.T(coef, -0.001, 1e3 * (tl.t1 + tl.t2) ** 2) - tl.Tp(-0.001)), 1e-9)

    print('P3 scheme mapping (defective coupling = scaled layer)')
    kk, th = (28.0, 0.8, 0.8), 90.0
    g = frob(kk) / knn_knt(kk, th)[0]
    td = twolayer_for_scheme(kk, th, 2.0e6, 0.004, 0.15, 1.5e6, 0.006, kc=frob(kk), left='N', right='N')
    cd = td.coefficients((340.0, 0.0), (305.0, 0.0))
    Teq_d = closed_pair_equilibrium(2.0e6 * 0.004, 1.5e6 * 0.006, 340.0, 305.0, g)
    ok('defective closed pair tends to (g Ma Ta + Mi Ti)/(g Ma + Mi)',
       float(td.T(cd, 0.0, 1e9)) - Teq_d, 1e-9)
    ok('interior diffusivity unchanged by the scaling', td.a1 - 0.8 / 2.0e6, 1e-20)

    print('extension: anisotropic circular inclusion')
    inc = AnisoInclusion(0.005, 28.0, 0.8, 1.0, 1000.0, 600.0, 300.0)
    worst_c, worst_f = 0.0, 0.0
    for ang in np.linspace(0, 2 * math.pi, 37)[:-1]:
        n = np.array([math.cos(ang), math.sin(ang)])
        eps = 1e-9
        xi, xo = inc.a * (1 - eps) * n, inc.a * (1 + eps) * n
        worst_c = max(worst_c, abs(float(inc.T(*xi)) - float(inc.T(*xo))))
        hh = 1e-7
        go = (float(inc.T(*(inc.a + 2 * hh) * n)) - float(inc.T(*(inc.a + hh) * n))) / hh
        fi = inc.k[0] * inc.g[0] * n[0] + inc.k[1] * inc.g[1] * n[1]
        worst_f = max(worst_f, abs(fi - inc.km * go) / 1000.0)
    ok('inclusion: temperature continuity on r=a', worst_c, 1e-6)
    ok('inclusion: co-normal flux continuity on r=a (FD)', worst_f, 1e-3)
    hh = 1e-5
    lap = sum(float(inc.T(0.008 + dx, 0.003 + dy)) for dx, dy in ((hh, 0), (-hh, 0), (0, hh), (0, -hh))) \
        - 4 * float(inc.T(0.008, 0.003))
    ok('inclusion: exterior field harmonic (FD)', lap / hh ** 2 * 1e-6, 1e-3)

    nfail = sum(1 for r in res if not r[3])
    print(f'\n{len(res)} checks, {nfail} failed')
    return res, nfail


if __name__ == '__main__':
    import sys
    _, nfail = selfcheck()
    sys.exit(1 if nfail else 0)
