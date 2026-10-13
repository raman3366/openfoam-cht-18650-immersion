#!/usr/bin/env python3
"""compare_smoke.py <mySmokeDir> <refSmokeDir> -- compare every number of result.json between my
re-run of the smoke instances and the design job's own run.  Prints, per case, the largest
absolute and relative difference over all numeric leaves (params, paths and wall time excluded)."""
import json, math, os, sys

SKIP = {'case', 'params', 'log'}


def leaves(d, pre=''):
    if isinstance(d, dict):
        for k, v in d.items():
            if pre == '' and k in SKIP:
                continue
            yield from leaves(v, pre + '.' + k if pre else k)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from leaves(v, f'{pre}[{i}]')
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield pre, float(d)


def main(mine, ref):
    rows, worst = [], 0.0
    names = sorted(n for n in os.listdir(ref) if os.path.isdir(os.path.join(ref, n)))
    for n in names:
        fr, fm = os.path.join(ref, n, 'result.json'), os.path.join(mine, n, 'result.json')
        if not os.path.exists(fr) and not os.path.exists(fm):
            fr2, fm2 = os.path.join(ref, n, 'result_mono.json'), os.path.join(mine, n, 'result_mono.json')
            if os.path.exists(fr2) and os.path.exists(fm2):
                fr, fm = fr2, fm2
            else:
                rows.append((n, 'no result.json in either (aborted in both?)' if not os.path.exists(fm2) else 'ref missing', '', '', ''))
                continue
        if not os.path.exists(fm) or not os.path.exists(fr):
            rows.append((n, 'MISSING in ' + ('mine' if not os.path.exists(fm) else 'ref'), '', '', ''))
            continue
        a, b = dict(leaves(json.load(open(fm)))), dict(leaves(json.load(open(fr))))
        dabs, drel, kabs, krel, nk = 0.0, 0.0, '', '', 0
        for k in a.keys() & b.keys():
            x, y = a[k], b[k]
            if math.isnan(x) and math.isnan(y):
                continue
            nk += 1
            d = abs(x - y)
            if d > dabs:
                dabs, kabs = d, k
            # relative difference only where the quantity is not itself round-off (|value| > 1e-6)
            if max(abs(x), abs(y)) > 1e-6:
                r = d / max(abs(x), abs(y))
                if r > drel:
                    drel, krel = r, k
        only = (a.keys() ^ b.keys())
        rows.append((n, f'{nk} numbers', f'{dabs:.2e} ({kabs})', f'{drel:.2e} ({krel})', f'{len(only)} keys differ' if only else ''))
        worst = max(worst, drel)
    for r in rows:
        print(' | '.join(r))
    print(f'largest relative difference over all cases (quantities above 1e-6): {worst:.3e}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
