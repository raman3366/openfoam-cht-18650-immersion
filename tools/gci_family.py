#!/usr/bin/env python3
"""
Grid-convergence analysis of one mesh family (Celik et al. 2008, J. Fluids Eng. 130:078001).

    python3 tools/gci_family.py runs/r1x_C60 runs/r1c_C30 runs/r1m_C13 runs/r1f_C6 [--csv FILE] [--t 267.0]

Run directories are given from the coarsest to the finest mesh.  Cell counts are read
from each run's own mesh (constant/<region>/polyMesh/owner), target quantities from its
monitors, all at the same physical time (default 267.0 s, the last 0.5 s monitor mark
every run reaches).

For every consecutive triplet the script reports the convergence ratio
R = (f_fine - f_mid) / (f_mid - f_coarse) and classifies it BEFORE anything else:

    0 < R < 1   monotone convergence      -> observed order p, Richardson value, GCI
   -1 < R < 0   oscillatory convergence   -> p and GCI per Celik, marked (osc)
    |R| >= 1    divergence                -> no order, no extrapolation, no GCI

With four meshes a least-squares fit f = f0 + a*h^p over all of them is added.
"""
import argparse, csv, glob, os, re, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="run directories, coarsest first")
ap.add_argument("--t", type=float, default=267.0, help="time at which quantities are compared [s]")
ap.add_argument("--heat", type=float, default=18.2, help="cell heat rate [W] used for h")
ap.add_argument("--csv")
a = ap.parse_args()
FS = 1.25


def ncells(run):
    n = 0
    for r in ("fluid", "battery"):
        head = open(os.path.join(run, "constant", r, "polyMesh", "owner"), "rb").read(3000).decode("latin1")
        n += int(re.search(r"nCells:\s*(\d+)", head).group(1))
    return n


def series(run, region, name, fname, col=1):
    rows = []
    for pat in (f"{run}/postProcessing/{region}/{name}/*/{fname}", f"{run}/postProcessing/{name}/{region}/*/{fname}"):
        for p in sorted(glob.glob(pat)):
            for line in open(p):
                if line.startswith("#") or not line.strip():
                    continue
                c = line.split()
                rows.append((float(c[0]), float(c[col])))
    if not rows:
        raise FileNotFoundError(f"{run}: {region}/{name}")
    arr = np.array(sorted(set(rows)))
    return arr[:, 0], arr[:, 1]


def targets(run, t0):
    out = {}
    t, tmax = series(run, "battery", "cellTemperatureMax", "volFieldValue.dat")
    if t[-1] < t0 - 1e-6:
        sys.exit(f"{run} ends at t = {t[-1]} s, before the comparison time {t0} s")
    tm, tmean = series(run, "battery", "cellTemperatureMean", "volFieldValue.dat")
    tc, tcan = series(run, "fluid", "canSurfaceTemperature", "surfaceFieldValue.dat")
    tb, tbath = series(run, "fluid", "bathTemperatureMean", "volFieldValue.dat")
    tu, umax = series(run, "fluid", "plumeSpeed", "volFieldValue.dat")
    at = lambda tt, yy: float(np.interp(t0, tt, yy))
    out["core hot spot (degC)"] = at(t, tmax) - 273.15
    out["cell mean (degC)"] = at(tm, tmean) - 273.15
    out["can surface (degC)"] = at(tc, tcan) - 273.15
    out["core-to-can drop (K)"] = at(t, tmax) - at(tc, tcan)
    out["bath bulk (degC)"] = at(tb, tbath) - 273.15
    # h(t): heat leaving the cell (generation minus storage) over area and (can - bath) difference
    area = np.pi * 0.018 * 0.065 + 2 * np.pi * 0.009**2
    cap = 2700 * np.pi * 0.009**2 * 0.065 * 1050
    dT = np.convolve(np.gradient(tmean, tm), np.ones(11) / 11, mode="same")
    h = (a.heat - cap * dT) / (area * (np.interp(tm, tc, tcan) - np.interp(tm, tb, tbath)))
    out["h at 200 s (W/m2K)"] = float(np.interp(200.0, tm, h))
    out["peak speed at 200 s (m/s)"] = float(np.interp(200.0, tu, umax))
    return out


def triplet(f3, f2, f1, r32, r21):
    """f3 coarse, f2 medium, f1 fine; r = h_coarser / h_finer > 1."""
    e21, e32 = f2 - f1, f3 - f2
    if e32 == 0 or e21 == 0:
        return dict(R=np.nan, kind="no change", p=np.nan, ext=f1, ea=0.0, gci=0.0)
    R = e21 / e32
    if abs(R) >= 1:
        return dict(R=R, kind="DIVERGENT", p=np.nan, ext=np.nan, ea=100 * abs(e21 / f1), gci=np.nan)
    s = np.sign(e32 / e21)
    p = 1.0
    for _ in range(200):
        q = np.log((r21**p - s) / (r32**p - s))
        pn = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(pn - p) < 1e-12:
            break
        p = pn
    ext = (r21**p * f1 - f2) / (r21**p - 1)
    ea = abs((f1 - f2) / f1)
    return dict(R=R, kind="monotone" if R > 0 else "oscillatory", p=p, ext=ext, ea=100 * ea,
                gci=100 * FS * ea / (r21**p - 1))


runs = [r.rstrip("/") for r in a.runs]
N = [ncells(r) for r in runs]
hrel = [(N[-1] / n) ** (1 / 3) for n in N]          # representative spacing relative to the finest mesh
T = [targets(r, a.t) for r in runs]
names = [os.path.basename(r) for r in runs]

print("=" * 100)
print(f"Mesh family, coarsest to finest, quantities at t = {a.t} s")
for nm, n, h in zip(names, N, hrel):
    print(f"  {nm:<14s} {n:>9,d} cells   h/h_finest = {h:.3f}")
print("  refinement ratios: " + ", ".join(f"{hrel[i] / hrel[i + 1]:.3f}" for i in range(len(N) - 1)))
print("=" * 100)
rows = []
for q in T[0]:
    vals = [t[q] for t in T]
    print(f"\n{q}")
    print("  values: " + "   ".join(f"{nm} {v:.4f}" for nm, v in zip(names, vals)))
    print("  steps : " + "   ".join(f"{vals[i + 1] - vals[i]:+.4f}" for i in range(len(vals) - 1)))
    row = {"quantity": q, **{nm: v for nm, v in zip(names, vals)}}
    for i in range(len(vals) - 2):
        g = triplet(vals[i], vals[i + 1], vals[i + 2], hrel[i] / hrel[i + 1], hrel[i + 1] / hrel[i + 2])
        tag = f"{names[i]}/{names[i + 1]}/{names[i + 2]}"
        if g["kind"] == "DIVERGENT":
            print(f"  {tag}: R = {g['R']:+.3f}  DIVERGENT: no order, no extrapolation, no GCI")
        elif g["kind"] == "no change":
            print(f"  {tag}: no change between meshes")
        else:
            print(f"  {tag}: R = {g['R']:+.3f}  {g['kind']}  p = {g['p']:.2f}  extrapolated {g['ext']:.4f}  "
                  f"e_a = {g['ea']:.3f} %  GCI_fine = {g['gci']:.3f} %"
                  f"  (= +/-{abs(g['gci'] / 100 * vals[i + 2]):.3f} in the units of the quantity)")
        row.update({f"R[{tag}]": g["R"], f"kind[{tag}]": g["kind"], f"p[{tag}]": g["p"],
                    f"ext[{tag}]": g["ext"], f"GCI%[{tag}]": g["gci"]})
    if len(vals) >= 4:
        # least squares f = f0 + a*h^p over all meshes (scan p, linear solve for f0 and a)
        best = None
        hh = np.array(hrel)
        for p in np.linspace(0.3, 4.0, 3701):
            A = np.vstack([np.ones_like(hh), hh**p]).T
            coef, res, *_ = np.linalg.lstsq(A, np.array(vals), rcond=None)
            sse = float(np.sum((A @ coef - np.array(vals)) ** 2))
            if best is None or sse < best[0]:
                best = (sse, p, coef[0], coef[1])
        sse, p, f0, aa = best
        print(f"  least squares over {len(vals)} meshes: p = {p:.2f}  f0 = {f0:.4f}  rms misfit {np.sqrt(sse / len(vals)):.2e}"
              f"  finest minus f0 = {vals[-1] - f0:+.4f}")
        row.update({"lsq_p": p, "lsq_f0": f0, "lsq_rms": np.sqrt(sse / len(vals))})
    rows.append(row)

if a.csv:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
