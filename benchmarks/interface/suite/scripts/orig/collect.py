#!/usr/bin/env python3
"""
collect.py -- tabulate result.json of several cases and compute observed orders.

    collect.py table <case> [<case> ...]
    collect.py order <key> <case_coarse> <case_mid> <case_fine> [...]   refinement ratio 2 assumed
          key: e.g.  error.exact.all.L2   error.scheme.all.Linf   audit.drift
"""
import json, math, os, sys


def get(d, key):
    for k in key.split('.'):
        d = d[k]
    return d


def load(case):
    return json.load(open(os.path.join(case, 'result.json')))


def order(key, cases, ratio=2.0):
    vals = [abs(get(load(c), key)) for c in cases]
    out = []
    for a, b in zip(vals[:-1], vals[1:]):
        out.append(math.log(a / b) / math.log(ratio) if a > 0 and b > 0 else float('nan'))
    return vals, out


if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[1] == 'order':
        vals, p = order(sys.argv[2], sys.argv[3:])
        print(sys.argv[2])
        for c, v in zip(sys.argv[3:], vals):
            print(f'  {os.path.basename(c):28s} {v:.6e}')
        print('  observed order: ' + '  '.join(f'{x:.3f}' for x in p))
    elif len(sys.argv) > 2 and sys.argv[1] == 'table':
        print(f"{'case':30s} {'coupling':12s} {'Linf exact':>11s} {'Linf scheme':>11s} {'L2 exact':>11s} {'if imbalance':>13s}")
        for c in sys.argv[2:]:
            r = load(c)
            e = r['error']
            sch = e.get('scheme', {}).get('all', {}).get('Linf', float('nan'))
            i = r['interface']
            imb = i['recomputed_imbalance'] if r['problem'] == 'slab' else i['imbalance']
            print(f"{os.path.basename(c):30s} {r['params']['coupling']:12s} {e['exact']['all']['Linf']:11.3e} "
                  f"{sch:11.3e} {e['exact']['all']['L2']:11.3e} {imb:+13.3e}")
    else:
        sys.exit(__doc__)
