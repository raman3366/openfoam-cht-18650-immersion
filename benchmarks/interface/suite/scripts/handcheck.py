#!/usr/bin/env python3
"""handcheck.py -- an energy ratio computed without ofio.py, post.py or exact.py.

Reads only the `internalField nonuniform List<scalar> N(...)` of the temperature files of
the smoke case p4_sca_t90_o4 (uniform one-dimensional mesh, so every cell of a region has the
same volume) and forms  -dE_iso/dE_aniso = -(rhoI cpI Li dmean(T_iso)) / (rhoA cpA La dmean(T_aniso)).
Parameters are typed here from the case description in the design (not read from bench.json):
aniso 4 mm, rho 2000, cp 1000; iso 6 mm, rho 1000, cp 1500; end time 20 s."""
import os, re, sys

case = sys.argv[1]


def mean_T(path):
    txt = open(path).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(([^)]*)\)', txt)
    v = [float(x) for x in m.group(2).split()]
    assert len(v) == int(m.group(1))
    return sum(v) / len(v)


dTa = mean_T(os.path.join(case, '20/aniso/T')) - mean_T(os.path.join(case, '0/aniso/T'))
dTi = mean_T(os.path.join(case, '20/iso/T')) - mean_T(os.path.join(case, '0/iso/T'))
ratio = -(1000 * 1500 * 0.006 * dTi) / (2000 * 1000 * 0.004 * dTa)
pred = (28 ** 2 + 0.8 ** 2 + 0.8 ** 2) ** 0.5 / 0.8
print(f'mean temperature change: aniso {dTa:+.9f} K, iso {dTi:+.9f} K')
print(f'heat received by iso / heat released by aniso = {ratio:.10f}')
print(f'sqrt(28^2 + 0.8^2 + 0.8^2)/0.8                 = {pred:.10f}   relative difference {ratio / pred - 1:+.1e}')
