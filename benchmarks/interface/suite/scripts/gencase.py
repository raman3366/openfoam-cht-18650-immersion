#!/usr/bin/env python3
"""
gencase.py -- write a complete OpenFOAM v2606 case for one point of the
verification matrix (two solid regions, chtMultiRegionFoam, blockMesh).

    gencase.py new  <problem> <caseDir> key=value ...   write the case
    gencase.py init <caseDir>                           fill 0/<region>/T after meshing
    gencase.py show <problem>                           list parameters and defaults

problems
    patch     P1  steady strip, T = phi(xi) + B*eta, oblique tensor, tangential gradient
    slab      P3/P4  transient strip (two-layer), closed or fixed-temperature ends
    annulus   P2  steady concentric annulus, cylindrical orthotropy inside
    ring      P5  steady full annulus, CARTESIAN orthotropy inside (curved interface whose
                  normal sweeps every angle to the principal axes); exact K-harmonic solution
                  (added by the benchmark job, see exact.RingHarmonic)

Regions: 'aniso' (constAnIso) and 'iso' (constIso).  Interface patches are the
mappedWall pair aniso_to_iso / iso_to_aniso made by splitMeshRegions.
Run a case with:   nice -n 10 openfoam2606 -c <caseDir>/Allrun
"""
import json
import math
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact as ex          # noqa: E402
import ofio                 # noqa: E402

COMMON = dict(
    coupling='directional',   # scalar | directional | function | conormal
    kc=0.0,                   # coupling conductivity for coupling=function [W/m/K]
    implicit='no',            # yes: useImplicit true on both sides (assembled coupling)
    nouter=1,                 # nOuterCorrectors in the top-level system/fvSolution
    order='iso aniso',        # solve order of the two solid regions
    ki=1.0,                   # isotropic region conductivity
    rhoA=2700.0, cpA=1000.0,  # anisotropic region
    rhoI=1000.0, cpI=2000.0,  # isotropic region
    dt=0.05, nsteps=400, nwrite=2,
    tol=1e-14,                # linear-solver tolerance for h
    dz=0.001,                 # thickness of the single cell layer in z
    conormal_lib='',          # library with the coNormalInterfaceFlux function object
    # ---- added by the benchmark job
    cn_lib='',                # libcoNormalCoupling.so (derived BC, guard and audit function objects)
    guard='none',             # none | warn | error | strict : anisotropicCouplingGuard function object
    audit=0,                  # 1: stock wallHeatFlux and coNormalWallHeatFlux executed every time step
    cpI1=0.0,                 # iso region: Cp = cpI + cpI1*T (hPolynomial) when non-zero
    Rlayer=0.0,               # interface layer resistance [m2 K/W] (thicknessLayers/kappaLayers), both sides
    allow_leak=0,             # slab: allow a zeroGradient aniso end although Knt*B != 0 (P17 test)
    nnonorth=0,               # nNonOrthogonalCorrectors of the aniso region (P25 test)
)
STRIP = dict(
    k1=28.0, k2=0.8, k3=0.8,  # principal conductivities
    theta=45.0,               # principal axis 1 against the interface normal [deg]
    phi=0.0,                  # rigid rotation of the whole problem in the mesh frame [deg]
    La=0.01, Li=0.01, H=0.01, nxa=16, nxi=16, ny=16,
    B=500.0,                  # tangential temperature gradient [K/m]
)
DEFAULTS = dict(
    patch=dict(COMMON, **STRIP, T0=320.0, A=1000.0,
               ybc='dirichlet',      # dirichlet: exact values on the eta-walls; gradient: fixedGradient +-B
               init='uniform'),      # uniform (T0) | exact
    slab=dict(COMMON, **STRIP, xbc='NN',   # N: zeroGradient, D: fixed value; first letter aniso end
              TL=350.0, TR=300.0,
              ic='step',             # step: Ta0 / Ti0 ; mode: single eigenmode ; uniform: Ta0 everywhere
              Ta0=350.0, Ti0=300.0, mode_n=1, mode_amp=20.0),
    annulus=dict(COMMON, r1=0.004, r2=0.009, r3=0.015, sector=90.0,
                 kr=0.8, kth=0.8, kz=28.0, ko=0.15, T1=350.0, T3=300.0,
                 nra=10, nri=12, nth=16, init='uniform'),
)
DEFAULTS['slab'].update(B=0.0, ny=1, dt=0.02, nsteps=250, nwrite=5)
DEFAULTS['annulus'].update(ki=0.15, rhoA=1.0, rhoI=1.0, dt=0.05, nsteps=600)
DEFAULTS['patch'].update(rhoA=1.0, rhoI=1.0)        # pseudo-time march to the steady state
DEFAULTS['ring'] = dict(COMMON, r1=0.004, r2=0.009, r3=0.015,
                        kx=28.0, ky=0.8, kz=0.8, ko=1.0,
                        psi=0.0,             # principal axis x of K against the mesh x axis [deg]
                        T0=350.0, Ex=100.0, Ey=500.0,   # interior gradient along the principal axes [K/m]
                        aq=0.0,              # amplitude a of the quadratic part a (X^2/kx - Y^2/ky)
                        bcdata='exact',      # exact | harm2 (T0 + dT2 cos 2phi on r1, T0 on r3; no exact solution)
                        dT2=10.0,
                        nra=10, nri=12, nth=8, init='uniform')
DEFAULTS['ring'].update(ki=1.0, rhoA=1.0, rhoI=1.0, dt=0.05, nsteps=600)

METHOD = dict(scalar='solidThermo', directional='directionalSolidThermo',
              function='function', conormal='directionalSolidThermo',
              conormalbc='directionalSolidThermo',   # derived BC of libcoNormalCoupling on both sides
              scalarbc='solidThermo')                # derived BC with the scalar option: the guard must stop it
CN_BC = ('conormalbc', 'scalarbc')
CONORMAL = ('conormal', 'conormalbc')


def wfile(case, rel, txt):
    path = os.path.join(case, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(textwrap.dedent(txt).lstrip('\n'))


H = ofio.header


def frame(P):
    """Unit normal (out of 'aniso') and tangent of the strip interface in mesh axes."""
    c, s = math.cos(math.radians(P['phi'])), math.sin(math.radians(P['phi']))
    return np.array([c, s, 0.0]), np.array([-s, c, 0.0])


# =========================================================================== new


def new(problem, case, args):
    P = dict(DEFAULTS[problem])
    for a in args:
        key, val = a.split('=', 1)
        if key not in P:
            sys.exit(f'unknown parameter {key!r} for problem {problem}; try: gencase.py show {problem}')
        P[key] = type(P[key])(val) if not isinstance(P[key], str) else val
    P['problem'] = problem
    if P['coupling'] not in METHOD:
        sys.exit('coupling must be one of ' + ', '.join(METHOD))
    if problem in ('annulus', 'ring'):
        P['ki'] = P['ko']
    if problem == 'ring':
        P['sector'] = 360.0
    if (P['coupling'] in CN_BC or P['guard'] != 'none' or P['audit']) and not P['cn_lib']:
        sys.exit('this case needs cn_lib=<path to libcoNormalCoupling.so>')
    if problem == 'slab':
        Knt = ex.knn_knt((P['k1'], P['k2'], P['k3']), P['theta'])[1]
        if 'N' in P['xbc'][0] and abs(Knt * P['B']) > 0 and not P['allow_leak']:
            sys.exit('zeroGradient on the aniso end is not adiabatic when Knt*B != 0 '
                     '(the solid Laplacian adds (K n)_t.grad_t T on every boundary face); use xbc=D?')
    os.makedirs(os.path.join(case, '0'), exist_ok=True)

    # ------------------------------------------------------------------ mesh
    # Topology: nblk pairs of blocks (aniso, iso).  Vertex (i1, s, iz): i1 = 0,1,2 across
    # the two regions, s = station along the interface direction, iz = z level.
    dz = P['dz']
    if problem in ('patch', 'slab'):
        n, t = frame(P)
        c1 = (-P['La'], 0.0, P['Li'])
        stations, wrap = [0.0, P['H']], False
        xy = lambda i1, s: c1[i1] * n[:2] + stations[s] * t[:2]
        n1a, n1i, n2 = P['nxa'], P['nxi'], P['ny']
        names = dict(a_lo='xmin', i_hi='xmax', lo='ymin', hi='ymax')
        arc = None
    else:
        c1 = (P['r1'], P['r2'], P['r3'])
        wrap = abs(P['sector'] - 360.0) < 1e-9
        nseg = 4 if wrap else 1
        stations = [math.radians(P['sector']) * i / nseg for i in range(nseg + (0 if wrap else 1))]
        xy = lambda i1, s: np.array([c1[i1] * math.cos(stations[s]), c1[i1] * math.sin(stations[s])])
        n1a, n1i, n2 = P['nra'], P['nri'], P['nth']
        names = dict(a_lo='inner', i_hi='outer', lo='side0', hi='side1')
        dth = math.radians(P['sector']) / nseg
        arc = lambda i1, s: (c1[i1] * math.cos(stations[s] + dth / 2), c1[i1] * math.sin(stations[s] + dth / 2))
    ns = len(stations)
    nblk = ns if wrap else ns - 1
    vid = lambda i1, s, iz: i1 + 3 * (s % ns) + 3 * ns * iz
    verts = '\n'.join(f'    ({xy(i1, s)[0]:.17g} {xy(i1, s)[1]:.17g} {z:.17g})'
                      for z in (0.0, dz) for s in range(ns) for i1 in range(3))
    blocks, edges, fb = [], [], []
    pf = dict(a_lo=[], i_hi=[], lo=[], hi=[])
    for s in range(nblk):
        for i1, zone, n1 in ((0, 'aniso', n1a), (1, 'iso', n1i)):
            v = [vid(i1, s, 0), vid(i1 + 1, s, 0), vid(i1 + 1, s + 1, 0), vid(i1, s + 1, 0),
                 vid(i1, s, 1), vid(i1 + 1, s, 1), vid(i1 + 1, s + 1, 1), vid(i1, s + 1, 1)]
            blocks.append(f"    hex ({' '.join(map(str, v))}) {zone} ({n1} {n2} 1) simpleGrading (1 1 1)")
            fb.append(f'({v[0]} {v[3]} {v[2]} {v[1]}) ({v[4]} {v[5]} {v[6]} {v[7]})')
            if s == 0 and not wrap:
                pf['lo'].append(f'({v[0]} {v[1]} {v[5]} {v[4]})')
            if s == nblk - 1 and not wrap:
                pf['hi'].append(f'({v[3]} {v[7]} {v[6]} {v[2]})')
        pf['a_lo'].append(f'({vid(0, s, 0)} {vid(0, s, 1)} {vid(0, s + 1, 1)} {vid(0, s + 1, 0)})')
        pf['i_hi'].append(f'({vid(2, s, 0)} {vid(2, s + 1, 0)} {vid(2, s + 1, 1)} {vid(2, s, 1)})')
        if arc:
            for iz, z in ((0, 0.0), (1, dz)):
                for i1 in range(3):
                    m = arc(i1, s)
                    edges.append(f'    arc {vid(i1, s, iz)} {vid(i1, s + 1, iz)} ({m[0]:.17g} {m[1]:.17g} {z:.17g})')
    bnd = '\n'.join(f"    {names[key]} {{ type wall; faces ({' '.join(pf[key])}); }}"
                    for key in ('a_lo', 'i_hi', 'lo', 'hi') if pf[key])
    wfile(case, 'system/blockMeshDict', H('dictionary', 'blockMeshDict') + f"""
scale 1;
vertices
(
{verts}
);
blocks
(
""" + '\n'.join(blocks) + """
);
edges
(
""" + '\n'.join(edges) + """
);
boundary
(
""" + bnd + f"""
    frontAndBack {{ type empty; faces ({' '.join(fb)}); }}
);
""")
    P['patch_names'] = names

    # ------------------------------------------------------------------ constant
    wfile(case, 'constant/regionProperties', H('dictionary', 'regionProperties', 'constant') + f"""
regions
(
    fluid ()
    solid ({P['order']})
);
""")
    # chtMultiRegionFoam constructs the gravity mesh object unconditionally
    # (fluid/createFluidFields.H:35), so constant/g must exist without a fluid region.
    wfile(case, 'constant/g', H('uniformDimensionedVectorField', 'g', 'constant') + """
dimensions      [0 1 -2 0 0 0 0];
value           (0 0 0);
""")
    if problem == 'ring':
        a = math.radians(P['psi'])
        csys = f"""
    type            cartesian;
    origin          (0 0 0);
    rotation
    {{
        type        axes;
        e1          ({math.cos(a):.17g} {math.sin(a):.17g} 0);
        e3          (0 0 1);
    }}"""
        kappa = f"({P['kx']:.17g} {P['ky']:.17g} {P['kz']:.17g})"
    elif problem == 'annulus':
        csys = """
    type            cylindrical;
    origin          (0 0 0);
    rotation
    {
        type        axes;
        e3          (0 0 1);
        e1          (1 0 0);
    }"""
        kappa = f"({P['kr']:.17g} {P['kth']:.17g} {P['kz']:.17g})"
    else:
        a = math.radians(P['theta'] + P['phi'])
        csys = f"""
    type            cartesian;
    origin          (0 0 0);
    rotation
    {{
        type        axes;
        e1          ({math.cos(a):.17g} {math.sin(a):.17g} 0);
        e3          (0 0 1);
    }}"""
        kappa = f"({P['k1']:.17g} {P['k2']:.17g} {P['k3']:.17g})"
    wfile(case, 'constant/aniso/thermophysicalProperties',
          H('dictionary', 'thermophysicalProperties', 'constant/aniso') + f"""
thermoType
{{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constAnIso;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}}
coordinateSystem
{{{csys}
}}
mixture
{{
    specie          {{ molWeight 50; }}
    thermodynamics  {{ Hf 0; Sf 0; Cp {P['cpA']:.17g}; }}
    transport       {{ kappa {kappa}; }}
    equationOfState {{ rho {P['rhoA']:.17g}; }}
}}
""")
    if P['cpI1'] != 0.0:        # Cp = cpI + cpI1*T; the stock solid combination with hPolynomial
        iso_tt = 'transport       polynomial;\n    thermo          hPolynomial;'
        iso_th = f"Hf 0; Sf 0; CpCoeffs<8> ({P['cpI']:.17g} {P['cpI1']:.17g} 0 0 0 0 0 0);"
        iso_tr = f"kappaCoeffs<8> ({P['ki']:.17g} 0 0 0 0 0 0 0);"
    else:
        iso_tt = 'transport       constIso;\n    thermo          hConst;'
        iso_th = f"Hf 0; Sf 0; Cp {P['cpI']:.17g};"
        iso_tr = f"kappa {P['ki']:.17g};"
    wfile(case, 'constant/iso/thermophysicalProperties',
          H('dictionary', 'thermophysicalProperties', 'constant/iso') + f"""
thermoType
{{
    type            heSolidThermo;
    mixture         pureMixture;
    {iso_tt}
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}}
mixture
{{
    specie          {{ molWeight 50; }}
    thermodynamics  {{ {iso_th} }}
    transport       {{ {iso_tr} }}
    equationOfState {{ rho {P['rhoI']:.17g}; }}
}}
""")

    # ------------------------------------------------------------------ system
    conormal = ''
    if P['coupling'] == 'conormal':
        if not P['conormal_lib']:
            sys.exit('coupling=conormal needs conormal_lib=<path to libcoNormalInterface.so>')
        conormal = f"""
    coNormal
    {{
        type            coNormalInterfaceFlux;
        libs            ("{os.path.abspath(P['conormal_lib'])}");
        region          aniso;
        field           qTan;
        guard           true;
        errors          strict;
        executeControl  timeStep;
        writeControl    writeTime;
    }}"""
    libs = f'libs            ("{os.path.abspath(P["cn_lib"])}");' if P['cn_lib'] else ''
    if P['guard'] != 'none':
        strict = '\n        errors          strict;' if P['guard'] == 'strict' else ''
        conormal += f"""
    guard
    {{
        type            anisotropicCouplingGuard;
        action          {'warn' if P['guard'] == 'warn' else 'error'};{strict}
    }}"""
    audit_ctl = 'timeStep' if P['audit'] else 'writeTime'
    if P['audit']:
        for r_ in ('aniso', 'iso'):
            conormal += f"""
    cnq_{r_}
    {{
        type            coNormalWallHeatFlux;
        region          {r_};
        executeControl  timeStep;
        writeControl    writeTime;
        log             false;
    }}"""
    winterval = max(1, P['nsteps'] // max(1, P['nwrite']))
    wfile(case, 'system/controlDict', H('dictionary', 'controlDict', 'system') + f"""
application     chtMultiRegionFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {P['dt'] * P['nsteps']:.12g};
deltaT          {P['dt']:.17g};
writeControl    timeStep;
writeInterval   {winterval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  16;
writeCompression off;
timeFormat      general;
timePrecision   12;
runTimeModifiable false;
adjustTimeStep  no;
maxCo           1e9;
maxDi           1e9;
{libs}

functions
{{{conormal}
    // gradient of T with the region's own gradScheme: its patch value is what the
    // tensor Laplacian uses for the tangential part of the boundary flux
    gradT_aniso
    {{
        type            grad;
        libs            (fieldFunctionObjects);
        region          aniso;
        field           T;
        result          gradT;
        executeControl  writeTime;
        writeControl    writeTime;
    }}
    // stock heat-flux objects, kept to document what they report (scalar alpha)
    whf_aniso
    {{
        type            wallHeatFlux;
        libs            (fieldFunctionObjects);
        region          aniso;
        patches         (aniso_to_iso);
        executeControl  {audit_ctl};
        writeControl    writeTime;
        log             false;
    }}
    whf_iso
    {{
        type            wallHeatFlux;
        libs            (fieldFunctionObjects);
        region          iso;
        patches         (iso_to_aniso);
        executeControl  {audit_ctl};
        writeControl    writeTime;
        log             false;
    }}
}}
""")
    wfile(case, 'system/fvSchemes', H('dictionary', 'fvSchemes', 'system') + """
ddtSchemes {} gradSchemes {} divSchemes {} laplacianSchemes {}
interpolationSchemes {} snGradSchemes {}
""")
    # nOuterCorrectors is read ONLY from this top-level file (readPIMPLEControls.H:3-8)
    wfile(case, 'system/fvSolution', H('dictionary', 'fvSolution', 'system') + f"""
PIMPLE
{{
    nOuterCorrectors {P['nouter']};
}}
""")
    solver = ('solver PBiCGStab; preconditioner DIC;' if P['implicit'] == 'yes'
              else 'solver PCG; preconditioner DIC;')
    for r in ('aniso', 'iso'):
        wfile(case, f'system/{r}/fvSchemes', H('dictionary', 'fvSchemes', f'system/{r}') + """
ddtSchemes           { default Euler; }
gradSchemes          { default Gauss linear; }
divSchemes           { default none; }
laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
""")
        wfile(case, f'system/{r}/fvSolution', H('dictionary', 'fvSolution', f'system/{r}') + f"""
solvers
{{
    "h.*"
    {{
        {solver}
        tolerance       {P['tol']:g};
        relTol          0;
        maxIter         5000;
    }}
}}
PIMPLE
{{
    nNonOrthogonalCorrectors {P['nnonorth'] if r == 'aniso' else 0};
}}
""")
    wfile(case, 'Allrun', f"""
#!/bin/sh
# usage:  nice -n 10 openfoam2606 -c {os.path.abspath(case)}/Allrun      (serial, one core)
cd "${{0%/*}}" || exit 1
set -e
TOOLS="{HERE}"
blockMesh > log.blockMesh 2>&1
splitMeshRegions -cellZones -overwrite > log.splitMeshRegions 2>&1
python3 "$TOOLS/gencase.py" init . > log.init 2>&1
chtMultiRegionFoam > log.chtMultiRegionFoam 2>&1
python3 "$TOOLS/post.py" . > log.post 2>&1
echo "done: $(tail -n 1 log.post)"
""")
    os.chmod(os.path.join(case, 'Allrun'), 0o755)
    json.dump(P, open(os.path.join(case, 'bench.json'), 'w'), indent=1, sort_keys=True)
    print('wrote', case)


# =========================================================================== init


def exact_objects(P):
    """Exact-solution objects for a parameter set (shared with post.py)."""
    out = {}
    if P['problem'] == 'ring':
        rg = ex.RingHarmonic(P['r2'], P['kx'], P['ky'], P['ko'], P['T0'], P['Ex'], P['Ey'], P['aq'], P['psi'])
        k = (P['kx'], P['ky'], P['kz'])
        out.update(kind='ring', obj=rg, k=k, kc=(ex.frob(k) if P['coupling'] in ('scalar', 'scalarbc') else None))
        return out
    if P['problem'] == 'annulus':
        k = (P['kr'], P['kth'], P['kz'])
        kc = {'scalar': ex.frob(k), 'function': P['kc']}.get(P['coupling'], P['kr'])
        an = ex.Annulus(P['r1'], P['r2'], P['r3'], P['kr'], P['ko'], P['T1'], P['T3'])
        out.update(kind='annulus', obj=an, kc=kc, knn=P['kr'], exact=an.solve(), scheme=an.solve(kc=kc))
        return out
    k = (P['k1'], P['k2'], P['k3'])
    Knn, Knt = ex.knn_knt(k, P['theta'])
    kc = ex.coupling_kappa(METHOD[P['coupling']], k, P['theta'], P['kc'])
    out.update(k=k, knn=Knn, knt=Knt, kc=kc)
    if P['problem'] == 'patch':
        s = ex.StripSteady.from_gradient(k, P['theta'], P['ki'], P['La'], P['Li'], P['T0'], P['A'], P['B'])
        out.update(kind='patch', obj=s, exact=s.exact(),
                   scheme=s.scheme(kc, conormal=(P['coupling'] in CONORMAL)))
        return out
    Ca, Ci = P['rhoA'] * P['cpA'], P['rhoI'] * P['cpI']
    kw = dict(left=P['xbc'][0], right=P['xbc'][1], TL=P['TL'], TR=P['TR'])
    leak = bool(P.get('allow_leak'))        # no exact solution for the leaky closed box (P17 test)
    tl = ex.twolayer_for_scheme(k, P['theta'], Ca, P['La'], P['ki'], Ci, P['Li'],
                                qjump=0.0 if leak else Knt * P['B'], **kw)
    qs = Knt * P['B'] * (kc / Knn) if (P['coupling'] in CONORMAL and not leak) else 0.0
    ts = ex.twolayer_for_scheme(k, P['theta'], Ca, P['La'], P['ki'], Ci, P['Li'], kc=kc, qjump=qs, **kw)
    out.update(kind='slab', obj=tl, scheme_obj=ts, Ca=Ca, Ci=Ci)

    def ic_coef(o):
        if P['ic'] == 'step':
            return o.coefficients((P['Ta0'], 0.0), (P['Ti0'], 0.0))
        if P['ic'] == 'uniform':
            return o.coefficients((P['Ta0'], 0.0), (P['Ta0'], 0.0))
        if P['ic'] == 'mode':                       # single eigenmode of the EXACT problem
            return None
        raise ValueError(P['ic'])

    if P['ic'] == 'mode':
        n = P['mode_n'] - 1
        xs = np.linspace(-P['La'], P['Li'], 2001)
        scale = P['mode_amp'] / np.max(np.abs(tl.X(n, xs)))
        c = np.zeros_like(tl.w)
        c[n] = scale
        base = P['Ti0'] if tl.zero_mode else 0.0
        out['coef'] = (c, base)
        out['scheme_coef'] = None                   # not an eigenmode of the scheme problem
    else:
        out['coef'], out['scheme_coef'] = ic_coef(tl), ic_coef(ts)
    return out


def initial_T(P, E, xi, eta):
    """Initial temperature at points given in interface coordinates."""
    if P['problem'] == 'patch':
        s = E['obj']
        return s.T(E['exact'], xi, eta) if P['init'] == 'exact' else np.full(np.shape(xi), P['T0']) + 0.0 * eta
    tl = E['obj']
    if P['ic'] == 'mode':
        return tl.T(E['coef'], xi, 0.0) + P['B'] * eta
    if P['ic'] == 'uniform':
        return np.full(np.shape(xi), P['Ta0']) + P['B'] * eta
    return np.where(np.asarray(xi) <= 0, P['Ta0'], P['Ti0']) + P['B'] * eta


def init(case):
    P = json.load(open(os.path.join(case, 'bench.json')))
    E = exact_objects(P)
    names = P['patch_names']
    for r, other in (('aniso', 'iso'), ('iso', 'aniso')):
        mesh = ofio.Mesh(os.path.join(case, 'constant', r, 'polyMesh'))

        if P['problem'] == 'ring':
            rg = E['obj']
            if P['bcdata'] == 'exact':
                bc_val = lambda X: rg.T(X[:, 0], X[:, 1])
            else:                          # second-harmonic data, no exact solution (P16 test)
                def bc_val(X):
                    r_, ph = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0]) - math.radians(P['psi'])
                    return np.where(r_ < P['r2'], P['T0'] + P['dT2'] * np.cos(2 * ph), P['T0'])
            Tint = (rg.T(mesh.C[:, 0], mesh.C[:, 1]) if (P['init'] == 'exact' and P['bcdata'] == 'exact')
                    else np.full(mesh.nCells, P['T0']))
            fixed = {}
        elif P['problem'] == 'annulus':
            an = E['obj']
            rad = lambda X: np.hypot(X[:, 0], X[:, 1])
            Tint = (an.T(E['exact'], rad(mesh.C)) if P['init'] == 'exact'
                    else np.full(mesh.nCells, 0.5 * (P['T1'] + P['T3'])))
            bc_val = lambda X: an.T(E['exact'], rad(X))
            fixed = {names['a_lo']: P['T1'], names['i_hi']: P['T3']}
        else:
            n, t = frame(P)
            co = lambda X: (X @ n, X @ t)
            Tint = initial_T(P, E, *co(mesh.C))
            if P['problem'] == 'patch':
                s = E['obj']
                bc_val = lambda X: s.T(E['exact'], *co(X))
            else:
                bc_val = lambda X: np.where(co(X)[0] <= 0, P['TL'], P['TR']) + P['B'] * co(X)[1]

        entries = []
        for pname, pp in mesh.patches.items():
            g = mesh.patch(pname)
            if pp['type'] == 'empty':
                body = 'type empty;'
            elif pp['type'].startswith('mapped'):
                method = METHOD[P['coupling']] if r == 'aniso' else 'solidThermo'
                extra = ''
                if r == 'aniso' and method == 'directionalSolidThermo':
                    extra += '\n        alphaAni        aniAlpha;'
                if r == 'aniso' and method == 'function':
                    extra += (f"\n        kappaValue      constant {P['kc']:.17g};"
                              f"\n        alphaValue      constant {P['kc'] / P['cpA']:.17g};")
                if P['coupling'] == 'conormal':
                    extra += ('\n        qr              qTan;\n        qrNbr           none;' if r == 'aniso'
                              else '\n        qr              none;\n        qrNbr           qTan;')
                if P['Rlayer'] > 0:
                    extra += f"\n        thicknessLayers ({P['Rlayer']:.17g});\n        kappaLayers     (1);"
                if P['implicit'] == 'yes':
                    extra += '\n        useImplicit     true;'
                bctype = ('compressible::coNormalTemperatureCoupledMixed' if P['coupling'] in CN_BC
                          else 'compressible::turbulentTemperatureRadCoupledMixed')
                body = (f"type            {bctype};\n"
                        f"        Tnbr            T;\n        kappaMethod     {method};{extra}\n"
                        f"        value           {ofio.scalar_list(Tint[g['cells']])};")
            elif P['problem'] == 'ring':
                body = f"type fixedValue; value {ofio.scalar_list(bc_val(g['Cf']))};"
            elif P['problem'] == 'annulus':
                if pname in fixed:
                    body = f"type fixedValue; value uniform {fixed[pname]:.17g};"
                else:                      # theta = const planes: normal is a principal axis
                    body = 'type zeroGradient;'
            else:
                is_x = pname in (names['a_lo'], names['i_hi'])
                if P['problem'] == 'patch':
                    dirichlet = is_x or P['ybc'] == 'dirichlet'
                else:
                    dirichlet = is_x and P['xbc'][0 if pname == names['a_lo'] else 1] == 'D'
                if dirichlet:
                    body = f"type fixedValue; value {ofio.scalar_list(bc_val(g['Cf']))};"
                elif is_x:
                    body = 'type zeroGradient;'
                else:                      # eta-walls: exact tangential gradient
                    sign = -1.0 if pname == names['lo'] else 1.0
                    body = f"type fixedGradient; gradient uniform {sign * P['B']:.17g};"
            entries.append(f"    {pname}\n    {{\n        {body}\n    }}")
        os.makedirs(os.path.join(case, '0', r), exist_ok=True)
        with open(os.path.join(case, '0', r, 'T'), 'w') as fh:
            fh.write(H('volScalarField', 'T', f'0/{r}')
                     + 'dimensions      [0 0 0 1 0 0 0];\n'
                     + f'internalField   {ofio.scalar_list(Tint)};\n'
                     + 'boundaryField\n{\n' + '\n'.join(entries) + '\n}\n')
        pent = '\n'.join(f"    {pn} {{ type {'empty' if pp['type'] == 'empty' else 'calculated; value uniform 1e5'}; }}"
                         for pn, pp in mesh.patches.items())
        with open(os.path.join(case, '0', r, 'p'), 'w') as fh:
            fh.write(H('volScalarField', 'p', f'0/{r}') + 'dimensions      [1 -1 -2 0 0 0 0];\n'
                     'internalField   uniform 1e5;\nboundaryField\n{\n' + pent + '\n}\n')
        print(f'{r}: {mesh.nCells} cells, patches {list(mesh.patches)}')


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'new' and sys.argv[2] in DEFAULTS:
        new(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif len(sys.argv) == 3 and sys.argv[1] == 'init':
        init(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'show' and sys.argv[2] in DEFAULTS:
        for k_, v_ in sorted(DEFAULTS[sys.argv[2]].items()):
            print(f'  {k_:14s} {v_!r}')
    else:
        sys.exit(__doc__)
