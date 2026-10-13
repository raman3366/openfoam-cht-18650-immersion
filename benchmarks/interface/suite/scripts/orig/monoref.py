#!/usr/bin/env python3
"""
monoref.py -- single-region (monolithic) reference for a benchmark case:
the same mesh and materials solved by solidFoam with pureZoneMixture, i.e.
without any region coupling.  In v2606 the zone-mixture branch of
solidThermo::heatDiffusion (solidThermo.C:308-381) builds the face flux from
the harmonic mean of n.A.n plus the full, linearly interpolated tensor, in
deferred-correction form  fvm::laplacian(A,h) - fvc::laplacian(A,h) +
fvc::laplacian(kappa,T).  One face flux serves both cells, so the operator is
conservative by construction; the correction needs iterations per time step
(ncorr, the PIMPLE outer correctors of solidFoam).

    monoref.py new  <partitionedCase> [ncorr=20]   writes <partitionedCase>_mono
    monoref.py init <monoCase>
    monoref.py post <monoCase>
Run:  nice -n 10 openfoam2606 -c <partitionedCase>_mono/Allrun
"""
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gencase            # noqa: E402
import ofio               # noqa: E402

H = ofio.header


def new(src, ncorr=20):
    P = json.load(open(os.path.join(src, 'bench.json')))
    case = src.rstrip('/') + '_mono'
    os.makedirs(os.path.join(case, 'system'), exist_ok=True)
    os.makedirs(os.path.join(case, '0'), exist_ok=True)
    shutil.copy(os.path.join(src, 'system', 'blockMeshDict'), os.path.join(case, 'system', 'blockMeshDict'))
    P['mono_ncorr'] = ncorr
    json.dump(P, open(os.path.join(case, 'bench.json'), 'w'), indent=1, sort_keys=True)
    # material blocks: reuse the partitioned 'aniso' dictionary text for the coordinate system
    txt = open(os.path.join(src, 'constant', 'aniso', 'thermophysicalProperties')).read()
    csys = txt[txt.index('coordinateSystem'):txt.index('mixture\n{')]
    if P['problem'] == 'annulus':
        ka = f"({P['kr']:.17g} {P['kth']:.17g} {P['kz']:.17g})"
    else:
        ka = f"({P['k1']:.17g} {P['k2']:.17g} {P['k3']:.17g})"
    ki = f"({P['ki']:.17g} {P['ki']:.17g} {P['ki']:.17g})"

    def zone(name, kappa, cp, rho):
        return (f"    {name}\n    {{\n        specie          {{ molWeight 50; }}\n"
                f"        thermodynamics  {{ Hf 0; Sf 0; Cp {cp:.17g}; }}\n"
                f"        transport       {{ kappa {kappa}; }}\n"
                f"        equationOfState {{ rho {rho:.17g}; }}\n    }}\n")

    gencase.wfile(case, 'constant/thermophysicalProperties',
                  H('dictionary', 'thermophysicalProperties', 'constant') + """
thermoType
{
    type            heSolidThermo;
    mixture         pureZoneMixture;
    transport       constAnIso;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}
""" + csys + "mixture\n{\n" + zone('aniso', ka, P['cpA'], P['rhoA']) + zone('iso', ki, P['cpI'], P['rhoI']) + "}\n")
    winterval = max(1, P['nsteps'] // max(1, P['nwrite']))
    gencase.wfile(case, 'system/controlDict', H('dictionary', 'controlDict', 'system') + f"""
application     solidFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {P['dt'] * P['nsteps']:.12g};
deltaT          {P['dt']:.17g};
writeControl    timeStep;
writeInterval   {winterval};
writeFormat     ascii;
writePrecision  16;
timeFormat      general;
timePrecision   12;
runTimeModifiable false;
adjustTimeStep  no;
maxDi           1e9;
""")
    gencase.wfile(case, 'system/fvSchemes', H('dictionary', 'fvSchemes', 'system') + """
ddtSchemes           { default Euler; }
gradSchemes          { default Gauss linear; }
divSchemes           { default none; }
laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
""")
    gencase.wfile(case, 'system/fvSolution', H('dictionary', 'fvSolution', 'system') + f"""
solvers
{{
    "h.*"
    {{
        solver PCG; preconditioner DIC;
        tolerance       {P['tol']:g};
        relTol          0;
        maxIter         5000;
    }}
}}
PIMPLE
{{
    nOuterCorrectors {ncorr};
    nCorrectors      1;
    nNonOrthogonalCorrectors 0;
}}
""")
    gencase.wfile(case, 'Allrun', f"""
#!/bin/sh
cd "${{0%/*}}" || exit 1
set -e
TOOLS="{HERE}"
blockMesh > log.blockMesh 2>&1
python3 "$TOOLS/monoref.py" init . > log.init 2>&1
solidFoam > log.solidFoam 2>&1
python3 "$TOOLS/monoref.py" post . > log.post 2>&1
echo "done: $(tail -n 1 log.post)"
""")
    os.chmod(os.path.join(case, 'Allrun'), 0o755)
    print('wrote', case)


def _setup(case):
    P = json.load(open(os.path.join(case, 'bench.json')))
    E = gencase.exact_objects(P)
    mesh = ofio.Mesh(os.path.join(case, 'constant', 'polyMesh'))
    if P['problem'] == 'annulus':
        co = lambda X: (np.hypot(X[:, 0], X[:, 1]), np.zeros(len(X)))
        in_aniso = co(mesh.C)[0] < P['r2']
    else:
        n, t = gencase.frame(P)
        co = lambda X: (X @ n, X @ t)
        in_aniso = co(mesh.C)[0] < 0
    return P, E, mesh, co, in_aniso


def init(case):
    P, E, mesh, co, _ = _setup(case)
    names = P['patch_names']
    if P['problem'] == 'annulus':
        an = E['obj']
        Tint = np.full(mesh.nCells, 0.5 * (P['T1'] + P['T3']))
    else:
        Tint = gencase.initial_T(P, E, *co(mesh.C))
    ent = []
    for pname, pp in mesh.patches.items():
        g = mesh.patch(pname)
        a, b = co(g['Cf'])
        if pp['type'] == 'empty':
            body = 'type empty;'
        elif P['problem'] == 'annulus':
            fixed = {names['a_lo']: P['T1'], names['i_hi']: P['T3']}
            body = f"type fixedValue; value uniform {fixed[pname]:.17g};" if pname in fixed else 'type zeroGradient;'
        else:
            is_x = pname in (names['a_lo'], names['i_hi'])
            if P['problem'] == 'patch':
                dirichlet, val = is_x or P['ybc'] == 'dirichlet', E['obj'].T(E['exact'], a, b)
            else:
                dirichlet = is_x and P['xbc'][0 if pname == names['a_lo'] else 1] == 'D'
                val = np.where(a <= 0, P['TL'], P['TR']) + P['B'] * b
            if dirichlet:
                body = f"type fixedValue; value {ofio.scalar_list(val)};"
            elif is_x:
                body = 'type zeroGradient;'
            else:
                body = f"type fixedGradient; gradient uniform {(-1.0 if pname == names['lo'] else 1.0) * P['B']:.17g};"
        ent.append(f"    {pname}\n    {{\n        {body}\n    }}")
    with open(os.path.join(case, '0', 'T'), 'w') as fh:
        fh.write(H('volScalarField', 'T', '0') + 'dimensions      [0 0 0 1 0 0 0];\n'
                 + f'internalField   {ofio.scalar_list(Tint)};\nboundaryField\n{{\n' + '\n'.join(ent) + '\n}\n')
    pent = '\n'.join(f"    {pn} {{ type {'empty' if pp['type'] == 'empty' else 'calculated; value uniform 1e5'}; }}"
                     for pn, pp in mesh.patches.items())
    with open(os.path.join(case, '0', 'p'), 'w') as fh:
        fh.write(H('volScalarField', 'p', '0') + 'dimensions      [1 -1 -2 0 0 0 0];\n'
                 'internalField   uniform 1e5;\nboundaryField\n{\n' + pent + '\n}\n')
    print(f'{mesh.nCells} cells, patches {list(mesh.patches)}')


def post(case):
    P, E, mesh, co, in_aniso = _setup(case)
    times = ofio.time_dirs(case)
    tval, tname = times[-1]
    sizes = {k: v['n'] for k, v in mesh.patches.items()}
    rd = lambda tn: ofio.read_field(os.path.join(case, tn, 'T'), mesh.nCells, sizes)[0]
    T = rd(tname)
    a, b = co(mesh.C)
    if P['problem'] == 'annulus':
        Te = E['obj'].T(E['exact'], a)
    elif P['problem'] == 'patch':
        Te = E['obj'].T(E['exact'], a, b)
    else:
        Te = E['obj'].T(E['coef'], a, tval) + P['B'] * b
    err = T - Te
    V = mesh.V
    res = dict(case=os.path.abspath(case), time=tval, cells=int(mesh.nCells), ncorr=P['mono_ncorr'],
               Linf=float(np.max(np.abs(err))), L2=float(math.sqrt(np.sum(V * err ** 2) / V.sum())))
    if len(times) >= 2:
        res['change_since_previous_write'] = float(np.max(np.abs(T - rd(times[-2][1]))))
    rc = np.where(in_aniso, P['rhoA'] * P['cpA'], P['rhoI'] * P['cpI'])
    E0 = float(np.sum(rc * rd(times[0][1]) * V) / P['dz'])
    E1 = float(np.sum(rc * T * V) / P['dz'])
    res['energy_change_J_per_m'] = E1 - E0
    res['energy_initial_J_per_m'] = E0
    log = open(os.path.join(case, 'log.solidFoam')).read()
    res['h_solves'] = log.count('Solving for h')
    res['steps'] = log.count('\nTime = ')
    json.dump(res, open(os.path.join(case, 'result.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
    print(f"RESULT monolithic Linf_vs_exact={res['Linf']:.3e} L2={res['L2']:.3e} dE={res['energy_change_J_per_m']:+.3e}")


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'new':
        kw = dict(a.split('=') for a in sys.argv[3:])
        new(sys.argv[2], int(kw.get('ncorr', 20)))
    elif len(sys.argv) == 3 and sys.argv[1] == 'init':
        init(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'post':
        post(sys.argv[2])
    else:
        sys.exit(__doc__)
