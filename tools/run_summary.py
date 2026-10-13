#!/usr/bin/env python3
"""One row per full-discharge run: end-of-discharge temperatures, heat-transfer coefficient,
energy balance and the oscillation verdict.

    python3 tools/run_summary.py runs/r1c_A30 runs/r1m_A13 ... [--csv results/r1_runs/summary.csv]

Temperatures and the global balance come from tools/analyze.py (the project's own table);
the heat-transfer coefficient is evaluated here at exactly 200 s and 250 s from the monitors
(same definition as tools/gci_family.py), with the heat rate the case itself applies;
the fluid energy residual and the alternating-mode verdict from tools/pilots/pilot_metrics.py
(per-time-step instruments).  Both outputs are kept in
the run directory as analyze.txt and metrics.txt.
"""
import argparse, csv, json, os, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="*")
ap.add_argument("--csv")
ap.add_argument("--reuse", action="store_true")
a = ap.parse_args()
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def first(pat, text, default=""):
    m = re.search(pat, text, re.M)
    return m.group(1).strip() if m else default


def h_exact(case, t_target):
    """Storage-based h at exactly t_target [s] (definition of tools/gci_family.py), with the
    heat rate of the case itself."""
    import glob
    import numpy as np
    def ser(region, name, fname):
        rows = []
        for f in sorted(glob.glob(f"{case}/postProcessing/{region}/{name}/*/{fname}")):
            for line in open(f):
                if line.startswith("#") or not line.strip():
                    continue
                c = line.split(); rows.append((float(c[0]), float(c[1])))
        arr = np.array(sorted(set(rows)))
        return arr[:, 0], arr[:, 1]
    q = 18.2
    fo = os.path.join(case, "system", "battery", "fvOptions")
    if os.path.isfile(fo):
        m = re.search(r"^\s*h\s+\(\s*([0-9.eE+-]+)\s+[0-9.eE+-]+\s*\)\s*;", open(fo).read(), re.M)
        if m:
            q = float(m.group(1)) * 1.65407e-5
    try:
        tm, ym = ser("battery", "cellTemperatureMean", "volFieldValue.dat")
        tc, yc = ser("fluid", "canSurfaceTemperature", "surfaceFieldValue.dat")
        tb, yb = ser("fluid", "bathTemperatureMean", "volFieldValue.dat")
    except Exception:
        return ""
    area = np.pi * 0.018 * 0.065 + 2 * np.pi * 0.009**2
    cap = 2700 * np.pi * 0.009**2 * 0.065 * 1050
    dT = np.convolve(np.gradient(ym, tm), np.ones(11) / 11, mode="same")
    h = (q - cap * dT) / (area * (np.interp(tm, tc, yc) - np.interp(tm, tb, yb)))
    return f"{float(np.interp(t_target, tm, h)):.1f}"


def analyze(case, cache):
    if a.reuse and os.path.exists(cache):
        return open(cache).read()
    r = subprocess.run([sys.executable, os.path.join(repo, "tools/analyze.py"), "--case", case, "--no-plots"],
                       capture_output=True, text=True, cwd=repo)
    try:
        open(cache, "w").write(r.stdout)
    except OSError:
        pass
    return r.stdout


rows = []
if True:
    for case in a.cases:
        case = case.rstrip("/")
        log = os.path.join(case, "log.chtMultiRegionFoam")
        if not os.path.exists(log) and os.path.exists(log + ".zst"):
            subprocess.run(["zstd", "-dq", "-k", log + ".zst", "-o", log], check=False)
        t = analyze(os.path.abspath(case), os.path.join(case, "analyze.txt"))
        mt = os.path.join(case, "metrics.txt")
        if not (a.reuse and os.path.exists(mt)):
            r = subprocess.run([sys.executable, os.path.join(repo, "tools/pilots/pilot_metrics.py"), case],
                               capture_output=True, text=True)
            open(mt, "w").write(r.stdout)
        meta = json.load(open(os.path.join(case, "run.json"))) if os.path.exists(os.path.join(case, "run.json")) else {}
        rows.append({"run": os.path.basename(case), "t": t, "m": open(mt).read(), "meta": meta,
                     "hx": (h_exact(os.path.abspath(case), 200.0), h_exact(os.path.abspath(case), 250.0))})

out = []
for r in rows:
    t, m, meta = r["t"], r["m"], r["meta"]
    out.append({
        "run": r["run"],
        "mesh": meta.get("mesh", ""),
        "nCorr": meta.get("nCorrectors", ""), "nOuter": meta.get("nOuterCorrectors", ""),
        "maxCo": meta.get("maxCo", ""), "maxDt": meta.get("maxDeltaT", ""),
        "T_core": first(r"cell hot spot\s+([0-9.]+) degC", t),
        "T_mean": first(r"cell volume mean\s+([0-9.]+) degC", t),
        "T_can": first(r"can surface \(mean\)\s+([0-9.]+) degC", t),
        "dT_radial": first(r"core-to-surface drop\s+([0-9.]+) K", t),
        "T_bath": first(r"bath bulk\s+([0-9.]+) degC", t),
        "h_200": r["hx"][0], "h_250": r["hx"][1],
        "u_peak": first(r"peak plume speed\s+([0-9.]+) m/s", t),
        "E_gen_J": first(r"generated\s+([0-9.]+) J", t),
        "global_resid_pct": first(r"integration error\)\s+[-0-9.]+ J\s+\(\s*([-0-9.]+) %\)", t),
        "fluid_resid_pct": first(r"RESIDUAL, open-system basis.*?=\s*([+-]?[0-9.]+) % of interface energy", m),
        "verdict": first(r"RUN VERDICT:\s*(.*?)(?:\s{2,}\(|$)", m),
        "accept": first(r"ACCEPTANCE \(candidate settings\):\s*(\w+)", m),
        "steps": first(r"^\s*steps\s+(\d+)\s+t_end", m),
        "note": meta.get("note", ""),
    })

cols = ["run", "T_core", "T_mean", "T_can", "dT_radial", "T_bath", "h_200", "h_250", "u_peak",
        "global_resid_pct", "fluid_resid_pct", "verdict", "accept", "steps"]
w = [max(len(c), *(len(str(o[c])) for o in out)) for c in cols] if out else [len(c) for c in cols]
w[cols.index("verdict")] = min(w[cols.index("verdict")], 30)
print("  ".join(c.ljust(x) for c, x in zip(cols, w)))
for o in out:
    print("  ".join(str(o[c])[:x].ljust(x) for c, x in zip(cols, w)))
if a.csv:
    os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(out[0].keys()), lineterminator="\n")
        cw.writeheader(); cw.writerows(out)
