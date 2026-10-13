#!/usr/bin/env python3
"""
runpts.py -- fast serial driver for lists of benchmark points.

    runpts.py <outDir> <block> [--list] [--redo]

A block is a function of blocks.py returning [(id, problem, {args}), ...].  For each point:
generate the case in-process (gencase.new), take the region meshes from a cache keyed by the
blockMeshDict (blockMesh + splitMeshRegions run once per distinct mesh), write the fields
in-process (gencase.init), run chtMultiRegionFoam (one process, serial), post-process
in-process (post.analyse).  Steady problems are continued (startFrom latestTime, twice the
steps) until the change between the last two written fields is below STEADY_TOL, at most
MAXCONT times.  One line per case goes to <outDir>/runlog_<block>.txt.

Must run inside the OpenFOAM environment; the queue runner applies niceness 10.
The case directories are complete and can be re-run on their own:  <case>/Allrun
"""
import hashlib, json, os, re, shutil, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gencase, post, blocks        # noqa: E402

STEADY_TOL = 1e-9       # K, change between the last two written fields
MAXCONT = 4


def sh(cmd, cwd, log):
    with open(os.path.join(cwd, log), 'w') as fh:
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode


def mesh(case, cache):
    key = hashlib.sha1(open(os.path.join(case, 'system/blockMeshDict'), 'rb').read()).hexdigest()[:16]
    cdir = os.path.join(cache, key)
    if not os.path.exists(os.path.join(cdir, 'ok')):
        shutil.rmtree(cdir, ignore_errors=True)
        os.makedirs(os.path.join(cdir, 'system'))
        for f in ('blockMeshDict', 'controlDict', 'fvSchemes', 'fvSolution'):
            shutil.copy(os.path.join(case, 'system', f), os.path.join(cdir, 'system', f))
        # controlDict without function objects or libs for the mesh utilities
        txt = open(os.path.join(cdir, 'system/controlDict')).read()
        txt = txt[:txt.index('functions')]
        txt = re.sub(r'\nlibs[^\n]*', '', txt)
        open(os.path.join(cdir, 'system/controlDict'), 'w').write(txt)
        os.makedirs(os.path.join(cdir, 'constant'))
        shutil.copy(os.path.join(case, 'constant/regionProperties'), os.path.join(cdir, 'constant'))
        if sh(['blockMesh'], cdir, 'log.blockMesh') or sh(['splitMeshRegions', '-cellZones', '-overwrite'], cdir,
                                                          'log.splitMeshRegions'):
            raise RuntimeError('meshing failed in ' + cdir)
        open(os.path.join(cdir, 'ok'), 'w').write('ok\n')
    for r in ('aniso', 'iso'):
        dst = os.path.join(case, 'constant', r, 'polyMesh')
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(os.path.join(cdir, 'constant', r, 'polyMesh'), dst)
    for f in ('log.blockMesh', 'log.splitMeshRegions'):
        shutil.copy(os.path.join(cdir, f), os.path.join(case, f))


def record_abort(case, problem, P, rc):
    """Write aborted.json from the solver log(s) and remove a result.json of an earlier leg."""
    import glob
    txt = ''.join(open(f).read() for f in sorted(glob.glob(os.path.join(case, 'log.chtMultiRegionFoam*'))))
    m = re.search(r'FOAM FATAL (IO )?ERROR.*?\n(.*?)\n\s*(From |file: )', txt, re.S)
    info = dict(case=os.path.abspath(case), problem=problem, params=P, aborted=True, rc=rc,
                steps_completed=txt.count('\nTime = '),
                fatal_message=(m.group(2).strip() if m else txt[-600:]),
                guard_fired=('anisotropic' in txt.lower() and 'kappaMethod solidThermo' in txt))
    json.dump(info, open(os.path.join(case, 'aborted.json'), 'w'), indent=1)
    if os.path.exists(os.path.join(case, 'result.json')):
        os.rename(os.path.join(case, 'result.json'), os.path.join(case, 'result_first_leg_not_converged.json'))
    return info


def run_point(out, pid, problem, args, cache):
    case = os.path.join(out, pid)
    shutil.rmtree(case, ignore_errors=True)
    args = dict(args)
    maxcont = int(args.pop('_maxcont', MAXCONT))     # driver option, not a generator parameter
    stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        gencase.new(problem, case, [f'{k}={v}' for k, v in args.items()])
        mesh(case, cache)
        gencase.init(case)
    finally:
        sys.stdout = stdout
    rc = sh(['chtMultiRegionFoam'], case, 'log.chtMultiRegionFoam')
    P = json.load(open(os.path.join(case, 'bench.json')))
    expect_fail = args.get('coupling') == 'scalarbc' or args.get('guard') in ('error', 'strict')
    if rc != 0:
        info = record_abort(case, problem, P, rc)
        return f"rc={rc} ABORTED after {info['steps_completed']} steps ({'guard' if info['guard_fired'] else 'other'})" \
               + ('' if expect_fail else '  UNEXPECTED')
    res = post.analyse(case)
    ncont = 0
    while (problem in ('patch', 'annulus', 'ring') and ncont < maxcont
           and res.get('steady_change_since_previous_write', 0.0) > STEADY_TOL):
        ncont += 1
        cd = os.path.join(case, 'system/controlDict')
        txt = open(cd).read()
        tend = float(re.search(r'\nendTime\s+([^;]+);', txt).group(1))
        txt = re.sub(r'\nstartFrom\s+[^;]+;', '\nstartFrom       latestTime;', txt)
        txt = re.sub(r'\nendTime\s+[^;]+;', f'\nendTime         {2 * tend:.12g};', txt)
        open(cd, 'w').write(txt)
        P['nsteps'] = int(round(2 * tend / P['dt']))
        json.dump(P, open(os.path.join(case, 'bench.json'), 'w'), indent=1, sort_keys=True)
        rc = sh(['chtMultiRegionFoam'], case, f'log.chtMultiRegionFoam.cont{ncont}')
        if rc != 0:             # a run that diverges after its first leg is an aborted run, not a result
            info = record_abort(case, problem, P, rc)
            return f"rc={rc} ABORTED in continuation {ncont} after {info['steps_completed']} steps (other)  UNEXPECTED"
        res = post.analyse(case)
    with open(os.path.join(case, 'log.post'), 'w') as fh:
        fh.write(post.summary(res) + '\n')
    tail = post.summary(res).splitlines()[-1]
    sc = res.get('steady_change_since_previous_write')
    return f"rc=0 cont={ncont}" + (f" steady={sc:.1e}" if sc is not None else '') + ' ' + tail


if __name__ == '__main__':
    out, block = os.path.abspath(sys.argv[1]), sys.argv[2]
    pts = getattr(blocks, block)()
    if '--list' in sys.argv:
        for pid, prob, a in pts:
            print(pid, prob, ' '.join(f'{k}={v}' for k, v in a.items() if not k.startswith('_')))
        print(len(pts), 'points')
        sys.exit(0)
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, '_meshcache')
    log = open(os.path.join(out, f'runlog_{block}.txt'), 'a')
    for pid, prob, a in pts:
        case = os.path.join(out, pid)
        if '--redo' not in sys.argv and (os.path.exists(os.path.join(case, 'result.json'))
                                         or os.path.exists(os.path.join(case, 'aborted.json'))):
            continue
        t0 = time.time()
        try:
            msg = run_point(out, pid, prob, a, cache)
        except BaseException as e:           # includes SystemExit from the generator
            if isinstance(e, KeyboardInterrupt):
                raise
            msg = 'DRIVER ERROR ' + ''.join(traceback.format_exception_only(type(e), e)).strip()[:300]
        print(f"{time.strftime('%H:%M:%S')} {pid:36s} wall={time.time() - t0:6.1f}s {msg}", file=log, flush=True)
