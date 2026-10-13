#!/usr/bin/env python3
"""pilot_metrics.py -- deciding quantities of one finished pilot.

    python3 pilot_metrics.py cases/P0                       # a finished pilot
    python3 pilot_metrics.py cases/P0 --asrun <postProcessing directory of the reference run>
    python3 pilot_metrics.py --post <postProcessing dir> [--log log[.gz]]   # committed run
    python3 pilot_metrics.py --selftest                     # synthetic check of the budget arithmetic

Reads the monitors (postProcessing/) and the solver log; never runs anything
and writes nothing into the case (only --json, where you say).
Sections: settings seen in the log, dt history, odd-even alternation, growth of
the alternating mode, fluid energy budget, extrema, acceptance summary.
The thresholds are the constants below; PILOTS.md quotes them from this file.

Two input modes, detected from the files present:
  pilot  : per-step instruments written by the harness of make_pilots.py
  legacy : runs that have only the 0.5 s monitors and the per-step
           wallHeatFlux.dat (the one-corrector runs with the velocity limiter).

"""
import argparse
import gzip
import json
import math
import os
import re
import sys

import numpy as np

T0_DEFAULT = 308.15
TSTD = 298.15
G = 9.81
RHO_C = (1177.5502, -0.708)          # kg/m3   constant/fluid/thermophysicalProperties
CP_C = (824.66, 3.6)                 # J/kg/K
RHO_SOLID = 2700.0                   # kg/m3   constant/battery/thermophysicalProperties
Q_SOLID = 1.10032e6                  # W/m3    system/battery/fvOptions
INTERFACE = "fluid_to_battery"
TIME_DIGITS = 6                      # set from the data in load(): digits of the printed times
OUT = {}                             # everything printed also lands here (--json)

# ---- acceptance thresholds (stated in PILOTS.md, sections B and C) -------------
RHO1_MIN = -0.2          # N4/N5: lag-1 autocorrelation of the step increments
# A series whose alternating part is below its floor is reported as 'below floor'
# and not judged: the increments of pure noise have rho1 = -0.5 at any amplitude.
FLOOR_REL = 1e-6         # alternating part / windowed mean |x|
FLOOR_U = 1e-7           # m/s  absolute floor for velocities
FLOOR_T = 1e-7           # K    absolute floor for temperatures
P0_PROBE_AMP = 0.028     # m/s  alternating amplitude of probe-0 Ux as run: committed coarse run, median |Ux|
                         #      of the 0.5 s samples 0.0286 over the first 2 s (0.025-0.029 over the discharge);
                         #      a repeat of that run on 4 ranks gives 0.0282
AMP_U_MAX = 0.01 * P0_PROBE_AMP      # film-probe alternating amplitude: at most 1 % of P0's
ALT_Q_MAX = 1e-5         # Q_int alternating part / windowed mean (committed coarse run 1.5e-3, P0 1.4e-3)
PARITY_MAX = 0.60        # R2-10: probe U changes sign on at most 60 % of steps (noise: 50 %)
TMEAN_DROP_MAX = 1e-5    # K    N5: bath MEAN never below T0 - 1e-5 K
TMIN_DROP_MAX = 1e-3     # K    R2-10: LOCAL minimum never more than about 1 mK below T0
T_FROM_STEP = 13         # both judged from this step on: the start-up pressure adjustment moves
                         # local T by 1.5e-5 K through dpdt at step 3 and rings until step 12
UNSET = 1e100            # a probe that was not located is written as -VGREAT = -1e300
                         # (sampling/probes/probeModels/internalField/internalFieldProbe.txx:38)
# growth rule (section B): see growth_phases()
GROW_R = -1.1            # every ratio of the run is below this (sign change and growth)
GROW_N, GROW_RISE = 8, 100.0         # at least 8 ratios with |d2| rising 100-fold, or
GROW_N_FAST, GROW_RISE_FAST = 4, 1e4  # at least 4 ratios with a 10,000-fold rise (fast divergence)
GROW_NEAR = 3            # phases of two series coincide if they overlap or lie within 3 steps
# pressure probes searched for a growth phase; the first gives the gain g compared between pilots
GROWTH_P_PROBES = (("fluidProbes", 1, "above the cell"), ("fluidProbes", 2, "below the cell"),
                   ("fluidProbes", 0, "film, mid-height"))
# As-run pilots (group A): on the x1.2 ramp the gain rises with dt on every step, so the median over
# a phase depends on how long the phase is.  Between such pilots the gain is compared on the same
# steps, all of them before any first dt cut at step 31 or later.
ASRUN_GAIN_STEPS = (22, 30)
# film probes judged by the acceptance summary: (function-object tag, column, description)
FILM_PROBES = (("fluidProbes", 0, "as-run probe 0, film mid-height (ON a cell-face plane)"),
               ("fluidProbesExtra", 7, "film, cell below the mid-height plane"),
               ("fluidProbesExtra", 8, "film, cell above the mid-height plane"),
               ("fluidProbesExtra", 5, "film near the base"),
               ("fluidProbesExtra", 6, "film near the top"))


def hs(T):
    """Sensible enthalpy of the fluid, J/kg (hPolynomial, Tstd 298.15 K; in
    v2606 icoPolynomial::H(p,T) = 0, so h depends on T only)."""
    return CP_C[0] * (T - TSTD) + 0.5 * CP_C[1] * (T * T - TSTD * TSTD)


def rho_f(T):
    return RHO_C[0] + RHO_C[1] * T


# ----------------------------------------------------------------------------
# readers
# ----------------------------------------------------------------------------
def find_fo(post, name):
    """Return the files of function object <name> (first time directory)."""
    hits = []
    for root, dirs, files in os.walk(post):
        if os.path.basename(root) == name:
            hits.append(root)
    out = {}
    for h in sorted(hits, key=len):
        for root, dirs, files in os.walk(h):
            for fn in files:
                out.setdefault(fn, os.path.join(root, fn))
    return out


_tok = re.compile(r"\(([^()]*)\)|(\S+)")


def read_table(path):
    """Generic function-object table: time, then scalar or (vector) columns.
    Returns t[n], list of arrays (n,) or (n,3).  Incomplete last rows dropped."""
    rows, ncol = [], None
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            vals = []
            try:
                for m in _tok.finditer(line):
                    if m.group(1) is not None:
                        vals.append([float(x) for x in m.group(1).split()])
                    else:
                        vals.append(float(m.group(2)))
            except ValueError:
                continue
            if ncol is None:
                ncol = len(vals)
            if len(vals) == ncol:
                rows.append(vals)
    if not rows:
        return np.zeros(0), []
    t = np.array([r[0] for r in rows])
    cols = []
    for k in range(1, ncol):
        cols.append(np.array([r[k] for r in rows], dtype=float))
    return t, cols


def read_wallheatflux(path):
    """One row per patch per time step: time patch min max integral."""
    t, data = [], {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            s = line.split()
            if len(s) != 5:
                continue
            try:
                tt, v = float(s[0]), [float(x) for x in s[2:5]]
            except ValueError:
                continue
            d = data.setdefault(s[1], [])
            d.append(v)
            if len(d) > len(t):
                t.append(tt)
    n = min(len(v) for v in data.values()) if data else 0
    return np.array(t[:n]), {k: np.array(v[:n]) for k, v in data.items()}


def read_header_value(path, key):
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*%s\s*:\s*(\S+)" % key, line)
            if m:
                return float(m.group(1))
    return None


def read_fieldminmax(path):
    """fieldMinMax with locations: time field min (x y z) [proc] max (x y z) [proc]."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            flat = re.sub(r"\([^()]*\)", "LOC", line).split()
            try:
                t, name = float(flat[0]), flat[1]
                nums = [float(x) for x in flat[2:] if x != "LOC"]
            except (ValueError, IndexError):
                continue
            locs = re.findall(r"\(([^()]*)\)", line)
            if len(nums) == 4:            # parallel: min proc max proc
                mn, mx = nums[0], nums[2]
            elif len(nums) == 2:
                mn, mx = nums
            else:
                continue
            out.setdefault(name, []).append((t, mn, mx, locs[0] if locs else "", locs[1] if len(locs) > 1 else ""))
    return out


_solve = re.compile(r"Solving for (\w+), Initial residual = (\S+), Final residual = (\S+), No Iterations (\d+)")
_lim = re.compile(r"Limited (\d+) \(([\d.eE+-]+)%\) of cells")


def parse_log(path, max_steps=None):
    """Per-time-step facts from a chtMultiRegionFoam log (plain or .gz)."""
    op = gzip.open if path.endswith(".gz") else open
    keys = ("time dt co_mean co_max di_max n_fluid n_solid n_Ux n_hf n_p p_it p_itmax p_r0 p_rl U_r0 U_rl "
            "h_r0 h_rl n_lim lim_first lim_last tmin tmax cont_local cont_global cont_cum clock exec").split()
    S = {k: [] for k in keys}
    cur = None
    pend = {}
    info = dict(nProcs=None, fvopt={}, relax={}, norelax={}, lookup={}, fieldrelax={}, end=False,
                fatal=[], warnings=0, warn_msgs=[], build=None, exe=None,
                probe_not_found=[], probe_multi=[], fo_lines=[])
    src = None
    ctx = None
    grab_warn = 0

    def flush():
        if cur is not None:
            for k in keys:
                S[k].append(cur.get(k, float("nan")))

    with op(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("Time = "):
                flush()
                if max_steps and len(S["time"]) >= max_steps:
                    cur = None
                    break
                cur = dict(time=float(line[7:]), n_fluid=0, n_solid=0, n_Ux=0, n_hf=0, n_p=0, p_it=0,
                           p_itmax=0, n_lim=0)
                cur.update(pend)
                pend = {}
                ctx = None
                continue
            if line.startswith("deltaT = "):
                pend["dt"] = float(line[9:])
                continue
            if line.startswith("Region: "):
                s = line.split()
                if "Courant" in line:
                    pend["co_mean"], pend["co_max"] = float(s[5]), float(s[7])
                elif "Diffusion" in line:
                    pend["di_max"] = float(s[7])
                continue
            if "Solving for" in line:
                if cur is None:
                    continue
                if line.startswith("Solving for fluid region"):
                    cur["n_fluid"] += 1
                    ctx = "fluid"
                    continue
                if line.startswith("Solving for solid region"):
                    cur["n_solid"] += 1
                    ctx = "solid"
                    continue
                m = _solve.search(line)
                if not m:
                    continue
                fld, r0, nit = m.group(1), float(m.group(2)), int(m.group(4))
                if fld == "p_rgh":
                    cur["n_p"] += 1
                    cur["p_it"] += nit
                    cur["p_itmax"] = max(cur["p_itmax"], nit)
                    cur.setdefault("p_r0", r0)
                    cur["p_rl"] = r0
                elif fld == "Ux":
                    cur["n_Ux"] += 1
                    cur.setdefault("U_r0", r0)       # first solve of the step
                    cur["U_rl"] = r0                 # last solve of the step (last outer pass)
                elif fld == "h" and ctx == "fluid":
                    cur["n_hf"] += 1
                    cur.setdefault("h_r0", r0)
                    cur["h_rl"] = r0
                continue
            if line.startswith("limitVelocity"):
                m = _lim.search(line)
                if m and cur is not None:
                    cur["n_lim"] += 1
                    cur.setdefault("lim_first", float(m.group(2)))
                    cur["lim_last"] = float(m.group(2))
                continue
            if line.startswith("Min/max T:"):
                if cur is not None and ctx == "fluid":
                    s = line[10:].split()
                    cur["tmin"], cur["tmax"] = float(s[0]), float(s[1])
                continue
            if line.startswith("time step continuity errors"):
                if cur is not None:
                    m = re.search(r"sum local = (\S+), global = (\S+), cumulative = (\S+)", line)
                    if m:
                        cur["cont_local"], cur["cont_global"], cur["cont_cum"] = (
                            float(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue
            if line.startswith("ExecutionTime"):
                if cur is not None:
                    s = line.split()
                    cur["exec"], cur["clock"] = float(s[2]), float(s[6])
                continue
            # ---- rare lines ----
            if "Did not find location" in line:
                # internalFieldProbe.C:112,121: the probe is skipped and written as -1e300
                if len(info["probe_not_found"]) < 20:
                    info["probe_not_found"].append(line.strip())
            elif "seems to be on multiple domains" in line:
                # internalFieldProbe.C:132,146
                if len(info["probe_multi"]) < 20:
                    info["probe_multi"].append(line.strip())
            elif "function object" in line.lower() and len(info["fo_lines"]) < 8:
                msg = line.strip()
                if msg not in info["fo_lines"]:
                    info["fo_lines"].append(msg)
            if line.startswith("Equation relaxation factor for "):
                s = line.split()
                k = "%s=%s" % (s[4], s[6])
                info["relax"][k] = info["relax"].get(k, 0) + 1
            elif line.startswith("No equation relaxation factor for "):
                k = line.split()[-1]
                info["norelax"][k] = info["norelax"].get(k, 0) + 1
            elif line.startswith("Lookup solver for "):
                k = line.split()[-1]
                info["lookup"][k] = info["lookup"].get(k, 0) + 1
            elif line.startswith("Field relaxation factor for "):
                s = line.split()
                k = "%s:%s" % (s[4], s[-1])
                info["fieldrelax"][k] = info["fieldrelax"].get(k, 0) + 1
            elif line.startswith("nProcs"):
                info["nProcs"] = int(line.split()[-1])
            elif line.startswith("Build"):
                info["build"] = line.split(":", 1)[1].strip()
            elif line.startswith("Exec"):
                info["exe"] = line.split(":", 1)[1].strip()
            elif "Source: " in line:
                src = line.split("Source:")[1].strip()
            elif "State: " in line and src:
                info["fvopt"][src] = line.split("State:")[1].strip()
                src = None
            elif line.startswith("End"):
                info["end"] = True
            elif line.startswith("trapFpe"):
                pass
            elif "FOAM FATAL" in line or "Floating point exception" in line or "sigFpe" in line \
                    or "MPI_ABORT" in line or "floating point exception" in line:
                if len(info["fatal"]) < 5:
                    info["fatal"].append(line.strip())
            elif "FOAM Warning" in line:
                info["warnings"] += 1
                grab_warn = 2
            elif grab_warn:
                grab_warn -= 1
                msg = line.strip()
                if msg and msg not in info["warn_msgs"] and len(info["warn_msgs"]) < 8:
                    info["warn_msgs"].append(msg)
        flush()
    A = {k: np.array(v, dtype=float) for k, v in S.items()}
    return A, info


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def d2(x):
    return x[2:] - 2.0 * x[1:-1] + x[:-2]


def d3(x):
    return x[3:] - 3.0 * x[2:-1] + 3.0 * x[1:-2] - x[:-3]


def alt_stats(x, window=21, floor_rel=FLOOR_REL, floor_abs=0.0):
    """Odd-even alternation of a per-step series.
    rho1  lag-1 autocorrelation of the step increments: -1 = pure 2-dt mode,
          +1 = smooth trend, -0.5 = white noise at ANY amplitude -- so rho1 is
          only judged for a series whose alternating part is above its floor
    zig   fraction of steps on which the second difference changes sign
          (100 % = zig-zag on every step, whatever the trend; white noise 73 %)
    alt   median over steps of |alternating part| / windowed mean of |x|;
          alternating part = |third difference|/8.  A 2-dt mode of amplitude A
          gives exactly A; a smooth trend gives only dt^3 (d3x/dt3)/8.  (The
          second difference/4 used before gives dt^2 (d2x/dt2)/4, which for a
          sqrt(t) heat rate at dt = 0.2 s is already 2e-5 of the mean, above
          the 1e-5 test.)
    amp   rms alternating part (units of x)
    below True when amp < max(floor_rel * mean windowed |x|, floor_abs):
          nothing alternates at a level that matters; reported, not judged"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return None
    d = np.diff(x)
    if np.ptp(d) <= 1e-14 * max(1e-300, np.max(np.abs(x))):
        return dict(rho1=float("nan"), zig=0.0, alt=0.0, altmax=0.0, amp=0.0, n=len(x), const=True,
                    below=True, floor=float(floor_abs))
    a, b = d[:-1] - d.mean(), d[1:] - d.mean()
    den = float(np.sum((d - d.mean()) ** 2))
    rho1 = float(np.sum(a * b) / den) if den > 0 else float("nan")
    dd = d2(x)
    zig = float(np.mean(dd[:-1] * dd[1:] < 0))
    alt = np.abs(d3(x)) / 8.0                      # alt[i] spans x[i..i+3]
    w = min(window, len(x))
    k = np.ones(w)
    wm = (np.convolve(np.abs(x), k, mode="same") / np.convolve(np.ones(len(x)), k, mode="same"))[1:-2]
    rel = alt / np.maximum(wm, 1e-300)
    amp = float(np.sqrt(np.mean(alt ** 2)))
    floor = max(floor_rel * float(np.mean(wm)), floor_abs)
    return dict(rho1=rho1, zig=zig, alt=float(np.median(rel)), altmax=float(np.max(rel)),
                amp=amp, n=len(x), const=False, below=bool(amp < floor), floor=float(floor))


def growth(x, a, b):
    """Per-step growth of the alternating mode over steps a..b (1-based):
    r[n] = d2[n+1]/d2[n]; a pure 2-dt mode of gain g gives r = -g."""
    x = np.asarray(x, dtype=float)
    dd = d2(x)                       # dd[i] is centred on step i+2
    i0, i1 = max(a - 2, 0), min(b - 2, len(dd) - 1)
    if i1 - i0 < 3:
        return None
    seg = dd[i0:i1 + 1]
    ok = (seg[:-1] != 0) & np.isfinite(seg[:-1]) & np.isfinite(seg[1:])
    r = seg[1:][ok] / seg[:-1][ok]
    if len(r) < 3:
        return None
    n = np.arange(len(seg))
    good = np.abs(seg) > 0
    slope = np.polyfit(n[good], np.log(np.abs(seg[good])), 1)[0] if good.sum() > 3 else float("nan")
    gm = float(np.exp(np.mean(np.log(np.abs(r[r != 0]))))) if np.any(r != 0) else 0.0
    return dict(steps=(a, b), ratios=r, median=float(np.median(r)), gmean=gm,
                neg=float(np.mean(r < 0)), fit=float(np.exp(slope)))


def growth_phases(x, first=3, last=None):
    """Coherent alternating phases of a per-step series, from the ratios
    r[n] = d2[n+1]/d2[n] of its second difference.
      grow  : the EARLIEST run of consecutive ratios r < -1.1 (sign change and
              growth on every step) that
                (a) has >= 8 ratios across which |d2| rises >= 100-fold, or
                    >= 4 ratios with a >= 10,000-fold rise (a fast divergence
                    leaves fewer than 8 steps), and
                (b) ends at the largest |d2| of the searched range so far.
              Why not simply 'a run of >= 4': any series whose second difference
              sits on a noise floor produces such runs by chance (white noise:
              in 83 % of 600-step series, with an apparent gain of 1.8; the
              committed coarse lid flux has 86).  (a) removes them.  (b) removes
              most recoveries of a stable, noise-driven 2-dt mode from a chance
              null of its envelope, which (a) alone accepts in 3-5 % of series.
      decay : the longest run of >= 4 consecutive ratios with -0.9 < r < 0 across
              which |d2| falls >= 10-fold
      tail  : share of negative ratios and geometric-mean |r| in the last third
      near  : runs of >= 4 ratios r < -1.1 that fail (a) or (b): count, longest
    Step numbers are 1-based and refer to the centre of the second difference.
    'first' is the first step searched (fixed-dt pilots: the step after the cap
    is reached, so that the varying dt of the ramp is not mixed in)."""
    x = np.asarray(x, dtype=float)
    dd = d2(x)                                   # dd[i] centred on step i+2
    last = len(dd) - 1 if last is None else min(last - 2, len(dd) - 1)
    i0 = max(first - 2, 0)
    if last - i0 < 2:
        return None, None, dict(neg=0.0, gmean=0.0, n=0, gain=1.0, steps=(first, first)), dict(count=0, longest=0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = dd[i0 + 1:last + 1] / dd[i0:last]
    r[~np.isfinite(r)] = 0.0
    mag = np.abs(dd[i0:last + 1])                # mag[k]: step k + i0 + 2;  |r[k]| = mag[k+1]/mag[k]
    mag[~np.isfinite(mag)] = np.inf

    def runs(mask, nmin):
        out, k = [], 0
        while k < len(mask):
            if mask[k]:
                j = k
                while j < len(mask) and mask[j]:
                    j += 1
                if j - k >= nmin:
                    out.append((k, j))
                k = j
            else:
                k += 1
        return out

    def rise(a, b):
        return float(mag[b] / mag[a]) if mag[a] > 0 else float("inf")

    def pack(a, b, n_runs):
        rr = np.abs(r[a:b])
        rr = rr[rr > 0]
        return dict(n=b - a, gmean=float(np.exp(np.mean(np.log(rr)))) if len(rr) else 0.0,
                    median=float(np.median(rr)) if len(rr) else 0.0,
                    steps=(a + i0 + 2, b + i0 + 2), ratios=r[a:b], count=n_runs,
                    amp0=float(mag[a]), amp1=float(mag[b]), rise=rise(a, b))

    good, weak = [], []
    for a, b in runs(r < GROW_R, 1):
        n, ri = b - a, rise(a, b)
        gate = (n >= GROW_N and ri >= GROW_RISE) or (n >= GROW_N_FAST and ri >= GROW_RISE_FAST)
        record = mag[b] >= np.max(mag[:a + 1])
        (good if gate and record else weak).append((a, b))
    grow = pack(good[0][0], good[0][1], len(good)) if good else None
    weak4 = [ab for ab in weak if ab[1] - ab[0] >= 4]
    near = dict(count=len(weak4), longest=max([b - a for a, b in weak4] + [0]))
    dr = [ab for ab in runs((r > -0.9) & (r < 0), 4) if rise(*ab) <= 0.1]
    decay = None
    if dr:
        a, b = max(dr, key=lambda ab: ab[1] - ab[0])
        decay = pack(a, b, len(dr))
    m = len(r) - len(r) // 3
    tail_r = r[m:]
    nz = tail_r[tail_r != 0]
    # Rise of |d2| over the last third, from the medians of its first and last quarters (the product
    # of the ratios is |d2_end/d2_start|, which one accidentally small value can inflate 100-fold).
    tm = mag[m:]
    tm = tm[np.isfinite(tm)]
    q = len(tm) // 4
    rise_q = float(np.median(tm[-q:]) / np.median(tm[:q])) if q >= 5 and np.median(tm[:q]) > 0 else 1.0
    tail = dict(neg=float(np.mean(tail_r < 0)) if len(tail_r) else 0.0,
                gmean=float(np.exp(np.mean(np.log(np.abs(nz))))) if len(nz) else 0.0, n=int(len(tail_r)),
                steps=(m + i0 + 2, len(r) + i0 + 2),
                # gain per step: the two quarters are centred 3/4 of the tail apart
                gain=float(rise_q ** (1.0 / (0.75 * len(tm)))) if len(tm) else 1.0)
    return grow, decay, tail, near


def slow_growth(tail, below=False):
    """A mode whose gain per step is between 1 and 1.1 never makes a run of ratios
    below -1.1, so the phase rule cannot see it.  It shows as a sustained
    alternation whose |d2| rises over the last third of the run; it counts as
    slow growth when that rise is at least the 100-fold asked of a phase."""
    n = tail.get("n", 0)
    if below or n < 20 or tail["neg"] < 0.9 or tail.get("gain", 1.0) <= 1.0:
        return False
    return n * math.log(tail["gain"]) >= math.log(GROW_RISE)


def verdict_growth(grow, decay, tail, below=False):
    """Verdict for ONE series.  'sustained' needs r < 0 on 90 % of the last third
    (white noise gives 73 %) and an alternating part above the floor."""
    sustained = tail["neg"] >= 0.9 and tail.get("n", 0) >= 20 and not below
    if grow is not None and sustained:
        return "GROWTH, then sustained 2-dt alternation"
    if grow is not None:
        return "GROWTH"
    if slow_growth(tail, below):
        return "SLOW GROWTH (sustained 2-dt alternation, |d2| x %.3f per step over the last third)" % tail["gain"]
    if sustained:
        return "SUSTAINED 2-dt alternation (no growth phase found)"
    if decay is not None:
        return "decay"
    return "no coherent alternating mode"


def coincident(p, q, near=GROW_NEAR):
    """Two growth phases coincide if their step ranges overlap or lie within 'near' steps."""
    return p["steps"][0] <= q["steps"][1] + near and q["steps"][0] <= p["steps"][1] + near


def combined_growth(res):
    """Run verdict from the per-series results (res: {label: dict(verdict=, phase=)}).
    Growth needs coincident phases in at least two series.  The phases of one
    real mode start at different steps (each series has its own trend and floor)
    but run to the same saturation or divergence, so the test is overlap of the
    step ranges, not equality of the onset step."""
    have = [(k, v["phase"]) for k, v in res.items() if v.get("phase")]
    n_series = len(res)
    best = []
    for i, (ka, pa) in enumerate(have):
        grp = [ka] + [kb for j, (kb, pb) in enumerate(have) if j != i and coincident(pa, pb)]
        if len(grp) > len(best):
            best = grp
    if len(best) >= 2:
        slow = all(res[k]["phase"].get("slow") for k in best)
        return ("GROWTH (slow: under %.1f per step)" % -GROW_R if slow else "GROWTH",
                "coincident growth phases in %d of %d series (%s)" % (len(best), n_series, ", ".join(best)))
    if have:
        return ("growth in ONE series only (not confirmed)",
                "%s shows a growth phase, no second series confirms it (%d series examined)"
                % (", ".join(k for k, _ in have), n_series))
    if any(v["verdict"].startswith("SUSTAINED") for v in res.values()):
        return "no growth; sustained 2-dt alternation", "no growth phase in any of %d series" % n_series
    if any(v["verdict"] == "decay" for v in res.values()):
        return "no growth; decay", "no growth phase in any of %d series" % n_series
    return "no growth; no coherent alternating mode", "no growth phase in any of %d series" % n_series


def P(msg=""):
    print(msg)


def head(title):
    P()
    P("== " + title + " " + "=" * max(0, 74 - len(title)))


def fmt_stats(name, s):
    if s is None:
        return "  %-40s (too short or not recorded)" % name
    if s.get("const"):
        return "  %-40s constant series" % name
    return "  %-40s rho1 %+6.3f   zig-zag %5.1f %%   alt/mean %9.3e (max %9.3e)   amp %9.3e%s" % (
        name, s["rho1"], 100 * s["zig"], s["alt"], s["altmax"], s["amp"],
        "   BELOW FLOOR (%.1e)" % s["floor"] if s.get("below") else "")


# ----------------------------------------------------------------------------
# data assembly
# ----------------------------------------------------------------------------
class Data:
    pass


def load(post, mode_hint=None):
    D = Data()
    D.post = post
    whf = find_fo(post, "wallHeatFlux").get("wallHeatFlux.dat")
    if not whf:
        sys.exit("no wallHeatFlux.dat under %s" % post)
    D.t, D.patch = read_wallheatflux(whf)
    D.n = len(D.t)
    global TIME_DIGITS
    with open(whf) as f:
        toks = [ln.split()[0] for ln in f if not ln.startswith("#")][:4000]
    TIME_DIGITS = max([len(x.replace(".", "").replace("-", "").lstrip("0").split("e")[0]) for x in toks] + [6])
    D.dt = np.diff(np.concatenate([[0.0], D.t]))
    D.step = {}                       # per-step series
    D.s05 = {}                        # as-run 0.5 s samples
    D.files = {}
    D.unset = {}                      # probe columns rejected because they hold the unset value

    def scalar(name, store, key, fname=None):
        fs = find_fo(post, name)
        p = fs.get(fname) if fname else (fs.get("volFieldValue.dat") or fs.get("surfaceFieldValue.dat"))
        if p:
            t, c = read_table(p)
            if len(c):
                store[key] = (t, c[0])
                D.files[key] = p
        return p

    for name in ("cellTemperatureMax", "cellTemperatureMean", "bathTemperatureMean", "bathTemperatureMax",
                 "plumeSpeed", "canSurfaceTemperature"):
        scalar(name, D.s05, name)
        scalar("step_" + name, D.step, name)
    for name in ("fluidTmin", "fluidMass", "fluidRhoH", "fluidH", "fluidDpdt", "fluidRhoMagSqrU",
                 "fluidRhoU", "cellH"):
        scalar("step_" + name, D.step, name)
    D.pilot = "fluidRhoH" in D.step
    for tag, name, store in (("fluidProbes", "fluidProbes", D.s05), ("batteryProbes", "batteryProbes", D.s05),
                             ("fluidProbes", "step_fluidProbes", D.step),
                             ("batteryProbes", "step_batteryProbes", D.step),
                             ("fluidProbesExtra", "step_fluidProbesExtra", D.step)):
        fs = find_fo(post, name)
        for fld in ("T", "U", "p_rgh"):
            if fld in fs:
                t, c = read_table(fs[fld])
                if len(c):
                    # a probe that was not located is written as -1e300 on every row
                    bad = [k for k, col in enumerate(c) if np.any(np.abs(col) > UNSET)]
                    for k in bad:
                        c[k] = None
                    if bad:
                        D.unset["%s.%s" % (name, fld)] = bad
                    store["%s.%s" % (tag, fld)] = (t, c)
    fs = find_fo(post, "step_fieldMinMax")
    D.minmax = read_fieldminmax(fs["fieldMinMax.dat"]) if "fieldMinMax.dat" in fs else {}
    fs = find_fo(post, "step_timeInfo")
    D.timeinfo = read_table(fs["timeInfo.dat"]) if "timeInfo.dat" in fs else None
    vf = D.files.get("bathTemperatureMean") or find_fo(post, "bathTemperatureMean").get("volFieldValue.dat")
    D.vol_fluid = read_header_value(vf, "Volume") if vf else None
    vs = find_fo(post, "step_cellH").get("volFieldValue.dat") or \
        find_fo(post, "cellTemperatureMean").get("volFieldValue.dat")
    D.vol_solid = read_header_value(vs, "Volume") if vs else None
    return D


def _align(D, t, v):
    """Put a function-object series on the wallHeatFlux steps (NaN where absent)."""
    if len(t) == D.n:
        return v
    idx = np.searchsorted(D.t, t - 1e-12 * np.maximum(1.0, np.abs(t)))
    out = np.full((D.n,) + v.shape[1:], np.nan)
    ok = idx < D.n
    out[idx[ok]] = v[ok]
    return out


def on_steps(D, key):
    """Per-step series aligned with the wallHeatFlux steps (None if absent)."""
    if key not in D.step:
        return None
    t, v = D.step[key]
    return _align(D, t, v)


def probe_series(D, tag, k, fld="U"):
    """Per-step series of probe k of function object <tag>: None if the file is
    absent, the column does not exist, or the probe was not located (unset)."""
    key = "%s.%s" % (tag, fld)
    if key not in D.step:
        return None
    t, cols = D.step[key]
    if k >= len(cols) or cols[k] is None or len(t) < 2:
        return None
    return _align(D, t, cols[k])


def film_probes(D):
    """The judged film probes that were recorded: [(label, description, U[n,3])]."""
    out = []
    for tag, k, what in FILM_PROBES:
        u = probe_series(D, tag, k, "U")
        if u is not None:
            out.append(("%s %d" % (tag, k), what, u))
    return out


def worst_component(u):
    """Velocity component with the largest alternating amplitude: (index, stats)."""
    best = None
    for c in range(u.shape[1]):
        s = alt_stats(u[:, c], floor_abs=FLOOR_U)
        if s is not None and (best is None or s["amp"] > best[1]["amp"]):
            best = (c, s)
    return best


def sign_changes(v):
    """Share of steps on which v changes sign (R2-10's parity statistic: a 2-dt
    mode about zero gives 100 %, noise about zero 50 %, a one-signed flow 0 %)."""
    v = v[np.isfinite(v)]
    return float(np.mean(v[1:] * v[:-1] < 0)) if len(v) > 1 else float("nan")


# ----------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------
def sec_log(L, info, meta):
    head("1. Settings as seen in the solver log")
    if L is None:
        P("  no log given: solve counts cannot be checked")
        return
    n = len(L["time"])
    P("  steps in log %d   nProcs %s   exec '%s'   finished with End: %s" % (n, info["nProcs"], info["exe"], info["end"]))
    if info["fatal"]:
        P("  FATAL / exception lines: " + " | ".join(info["fatal"][:3]))
    if info["warnings"]:
        P("  FOAM warnings: %d   first messages: %s" % (info["warnings"], " | ".join(info["warn_msgs"][:4])))
    for key, what in (("probe_not_found", "probe NOT LOCATED ('Did not find location')"),
                      ("probe_multi", "probe on several domains ('seems to be on multiple domains')")):
        for msg in info[key][:6]:
            P("  %s: %s" % (what, msg))
    if not info["probe_not_found"] and not info["probe_multi"]:
        P("  probe location warnings in the log: none")
    for msg in info["fo_lines"]:
        P("  function-object message: %s" % msg)
    ex = (meta or {}).get("expect", {})
    res = {}

    def row(label, arr, want):
        arr = arr[np.isfinite(arr)]
        lo, med, hi = (int(arr.min()), float(np.median(arr)), int(arr.max())) if len(arr) else (0, 0, 0)
        ok = "" if want is None else ("  expected %d  -> %s" % (want, "OK" if lo == hi == want else "MISMATCH"))
        P("  %-44s min %d  median %g  max %d%s" % (label, lo, med, hi, ok))
        res[label] = dict(min=lo, median=med, max=hi, expected=want)

    row("fluid-region solves per time step", L["n_fluid"], ex.get("fluidSolvesPerStep"))
    row("p_rgh linear solves per time step", L["n_p"], ex.get("pRghSolvesPerStep"))
    row("momentum (Ux) solves per time step", L["n_Ux"], ex.get("UxSolvesPerStep"))
    row("fluid energy (h) solves per time step", L["n_hf"], ex.get("fluidSolvesPerStep"))
    row("limitVelocity lines per time step", L["n_lim"], ex.get("limiterLinesPerStep"))
    row("solid-region solves per time step", L["n_solid"], ex.get("fluidSolvesPerStep"))
    for src, st in sorted(info["fvopt"].items()):
        want = None
        if ex:
            want = {"limitU": "active" if ex.get("limiter") else "inactive",
                    "heatSource": "active" if ex.get("heatSource") else "inactive"}.get(src)
        P("  fvOption %-12s state %-9s%s" % (src, st, "" if want is None else
                                             "  expected %s -> %s" % (want, "OK" if want == st else "MISMATCH")))
    if info["relax"] or info["norelax"]:
        P("  relaxation factors actually applied (count over the run):")
        for k, v in sorted(info["relax"].items()):
            P("      %-22s x %d" % (k, v))
        for k, v in sorted(info["norelax"].items()):
            P("      %-22s x %d   (no factor: relax() skipped)" % (k + " none", v))
        if ex:
            nst, no = n, ex.get("nOuter", 1)
            want_final = nst if ex.get("relaxFinal", True) else 0
            got_final = info["relax"].get("UFinal=1", 0)
            got_nofinal = info["norelax"].get("UFinal", 0)
            got_07 = info["relax"].get("U=0.7", 0)
            P("      check: UFinal=1 %d (expected %d), UFinal none %d (expected %d), U=0.7 %d (expected %d) -> %s"
              % (got_final, want_final, got_nofinal, nst - want_final, got_07, nst * (no - 1),
                 "OK" if (got_final == want_final and got_nofinal == nst - want_final and got_07 == nst * (no - 1))
                 else "MISMATCH"))
    else:
        P("  relaxation-factor lines: none in this log (DebugSwitches{solution 1} not active)")
    if info["lookup"]:
        P("  solver dictionaries selected: " + ", ".join("%s x %d" % kv for kv in sorted(info["lookup"].items())))
    lim = L["lim_first"]
    if np.isfinite(lim).any():
        P("  limiter, cells clipped after the momentum predictor: median %.2f %%  max %.2f %%;"
          " after the last corrector: median %.2f %%  max %.2f %%"
          % (np.nanmedian(L["lim_first"]), np.nanmax(L["lim_first"]), np.nanmedian(L["lim_last"]), np.nanmax(L["lim_last"])))
    pr = L["p_rl"]
    if np.isfinite(pr).any():
        P("  p_rgh initial residual, first solve: median %.3g  max %.3g;  last solve: median %.3g  max %.3g"
          % (np.nanmedian(L["p_r0"]), np.nanmax(L["p_r0"]), np.nanmedian(pr), np.nanmax(pr)))
        P("  p_rgh iterations per step: median %g  max single solve %d  (solves at the 1000 cap: %d steps)"
          % (np.nanmedian(L["p_it"]), int(np.nanmax(L["p_itmax"])), int(np.sum(L["p_itmax"] >= 1000))))
    for key, name in (("U", "Ux"), ("h", "fluid h")):
        r0, rl = L[key + "_r0"], L[key + "_rl"]
        if np.isfinite(rl).any():
            P("  %s initial residual, first solve of the step: median %.3g  max %.3g;  last solve (last outer pass): median %.3g  max %.3g"
              % (name, np.nanmedian(r0), np.nanmax(r0), np.nanmedian(rl), np.nanmax(rl)))
    if np.isfinite(L["co_max"]).any():
        P("  Courant max: median %.3g  max %.3g    continuity cumulative at end: %.3e"
          % (np.nanmedian(L["co_max"]), np.nanmax(L["co_max"]), L["cont_cum"][-1] if n else float("nan")))
    if np.isfinite(L["clock"]).any() and n > 1:
        P("  wall clock %.0f s for %d steps = %.3f s/step (log ClockTime, whole seconds)" % (L["clock"][-1], n, L["clock"][-1] / n))
        OUT["s_per_step"] = float(L["clock"][-1] / n)
    OUT["log"] = dict(steps=n, end=info["end"], fatal=info["fatal"], counts=res, fvopt=info["fvopt"],
                      relax=info["relax"], norelax=info["norelax"], probe_not_found=info["probe_not_found"],
                      probe_multi=info["probe_multi"], fo_lines=info["fo_lines"],
                      p_cap_steps=int(np.sum(L["p_itmax"] >= 1000)) if n else 0)


def first_cut(dt):
    for n in range(1, len(dt)):
        if dt[n] < dt[n - 1] * (1.0 - 1e-6):
            return n + 1                 # 1-based step whose dt is smaller than the previous one
    return None


def sec_dt(D, L):
    head("2. Time-step history")
    dt, t = D.dt, D.t
    if D.timeinfo is not None and len(D.timeinfo[1]) >= 4 and len(D.timeinfo[1][3]) > 3:
        # timeInfo columns: cpuTime clockTime cpu/step clock/step (timeInfo.C:108-121).  The clock is in
        # whole seconds, so the per-step cost is taken from the cpu time of the master rank and scaled
        # by the run's overall clock/cpu ratio.
        cpu_tot, clk_tot = D.timeinfo[1][0], D.timeinfo[1][1]
        c = D.timeinfo[1][2][1:]                    # cpu per step; the first value includes start-up
        scale = clk_tot[-1] / cpu_tot[-1] if cpu_tot[-1] > 0 else 1.0
        med = float(np.median(c))
        P("  cost per time step (timeInfo, master-rank cpu): median %.3f s, max %.3f s; wall clock = %.2f x cpu over the run,"
          " so about %.2f s of wall clock per normal step" % (med, c.max(), scale, scale * med))
        OUT["s_per_step_timeInfo"] = float(scale * med)
        slow = np.where(c > 5.0 * med)[0] + 2                     # 1-based step numbers
        if len(slow):
            extra = float(scale * (c[slow - 2].sum() - len(slow) * med))
            P("      %d step(s) cost more than 5 x the median (steps %s%s): median %.1f s of cpu each; together %.0f s of wall"
              % (len(slow), " ".join(str(k) for k in slow[:12]), " ..." if len(slow) > 12 else "",
                 np.median(c[slow - 2]), extra))
            P("      clock more than normal steps would cost (%.0f %% of the run).  A p_rgh solve that runs to its"
              " 1000-iteration limit does this: see the count in section 1." % (100 * extra / max(scale * c.sum(), 1e-300)))
            OUT["slow_steps_extra_s"] = extra
        else:
            P("      no step cost more than 5 x the median")
        OUT["slow_steps"] = [int(k) for k in slow]
    P("  steps %d   t_end %.10g s   dt first %.6g   max %.6g   median %.6g   last %.6g"
      % (D.n, t[-1], dt[0], dt.max(), np.median(dt), dt[-1]))
    r = dt[1:] / dt[:-1]
    n12 = 0
    while n12 < len(r) and abs(r[n12] - 1.2) < 2e-3:
        n12 += 1
    P("  x1.2 growth holds for steps 2..%d (dt reaches %.6g s at t = %.6g s)" % (n12 + 1, dt[n12], t[n12]))
    fc = first_cut(dt)
    OUT["dt"] = dict(steps=int(D.n), t_end=float(t[-1]), ramp_last_step=int(n12 + 1), first_cut=fc,
                     dt_max=float(dt.max()), dt_median=float(np.median(dt)))
    if fc is None:
        P("  dt is never cut (monotone ramp, then constant): no Courant or write-time feedback")
    else:
        P("  first dt cut: step %d, t = %.6g s, dt %.4g -> %.4g s (factor %.3f)"
          % (fc, t[fc - 1], dt[fc - 2], dt[fc - 1], dt[fc - 1] / dt[fc - 2]))
        OUT["dt"].update(cut_from=float(dt[fc - 2]), cut_to=float(dt[fc - 1]), cut_time=float(t[fc - 1]))
        if L is not None and len(L["co_max"]) >= fc and np.isfinite(L["co_max"][fc - 1]):
            P("      Courant max entering that step (log): %.3g" % L["co_max"][fc - 1])
    for a, b in ((0.5, 1.0), (1.0, 2.0), (2.0, 10.0), (10.0, 1e9)):
        m = (t > a) & (t <= b)
        if m.sum() > 2:
            P("  dt in %g-%g s: median %.4g  min %.4g  max %.4g  (%d steps)" % (a, min(b, t[-1]), np.median(dt[m]), dt[m].min(), dt[m].max(), m.sum()))
    for T in (0.5, 1.0, 2.0):
        if t[-1] >= T * (1 - 1e-9):
            P("  steps to t = %g s: %d" % (T, int(np.searchsorted(t, T * (1 - 1e-9)) + 1)))
    if L is not None and np.isfinite(L["dt"]).any():
        m = min(len(L["dt"]), D.n)
        # monitor times carry timePrecision digits, so compare where they resolve dt to better than 1e-3
        res = 0.5 * 10.0 ** (np.floor(np.log10(np.maximum(t[:m], 1e-300))) - (TIME_DIGITS - 1))
        ok = res / dt[:m] < 1e-3
        if ok.any():
            dev = np.nanmax(np.abs(L["dt"][:m][ok] - dt[:m][ok]) / dt[:m][ok])
            P("  log 'deltaT =' against monitor time differences: max relative difference %.2e over the %d steps the "
              "printed times resolve" % (dev, int(ok.sum())))


def sec_alt(D, n0):
    head("3. Odd-even (2-dt) alternation, steps %d..%d" % (n0, D.n))
    P("  rho1 = lag-1 autocorrelation of step increments; zig-zag = share of steps where the second difference")
    P("  changes sign; alt/mean = |third difference|/8 over the 21-step mean of |x|; amp = its rms (units of x).")
    P("  BELOW FLOOR = amp under max(%.0e of the mean |x|, the absolute floor): reported, not judged." % FLOOR_REL)
    res = {}
    sl = slice(n0 - 1, None)

    def put(label, v, kind=None, **kw):
        s = alt_stats(v[sl], **kw)
        P(fmt_stats(label, s))
        if s is not None:
            s["kind"] = kind
        res[label] = s
        return s

    put("Q_int = interface heat rate [W]", D.patch[INTERFACE][:, 2], "Q")
    if "bathLid" in D.patch:
        put("bathLid min heat flux [W/m2]", D.patch["bathLid"][:, 0], floor_rel=0.0, floor_abs=5.0 * FLOOR_T)
    for key, label, kind, kw in (
            ("bathTemperatureMean", "bath mean T [K]", "Tmean", dict(floor_rel=0.0, floor_abs=FLOOR_T)),
            ("plumeSpeed", "max|U| [m/s]", None, dict(floor_abs=FLOOR_U)),
            ("fluidTmin", "min T fluid [K]", None, dict(floor_rel=0.0, floor_abs=FLOOR_T)),
            ("fluidDpdt", "int(dpdt) dV [W]", None, {})):
        v = on_steps(D, key)
        if v is not None:
            put(label, v, kind, **kw)
    # film probes: judged on the velocity component that alternates most
    judged = set()
    for label, what, u in film_probes(D):
        judged.add(label)
        best = worst_component(u[sl])
        if best is None:
            P(fmt_stats("%s U" % label, None))
            continue
        c, s = best
        s.update(kind="film", comp="xyz"[c], parity=sign_changes(u[sl][:, c]),
                 medabs=float(np.nanmedian(np.abs(u[sl][:, c]))), what=what)
        name = "%s U%s [m/s]" % (label, "xyz"[c])
        P(fmt_stats(name, s))
        P("  %-40s sign changes on %.1f %% of steps; median |U%s| %.4g m/s   (%s)"
          % ("", 100 * s["parity"], "xyz"[c], s["medabs"], what))
        res[name] = s
    # the other probes: printed, not judged
    for tag in ("fluidProbes", "fluidProbesExtra"):
        if tag + ".U" in D.step:
            for p in range(len(D.step[tag + ".U"][1])):
                if "%s %d" % (tag, p) in judged:
                    continue
                u = probe_series(D, tag, p, "U")
                if u is None:
                    P("  %-40s NOT LOCATED by the solver (unset value in the file): rejected" % ("%s %d U" % (tag, p)))
                    continue
                comp = int(np.nanargmax(np.nanmean(np.abs(u[sl]), axis=0)))
                put("%s %d U%s [m/s]" % (tag, p, "xyz"[comp]), u[:, comp], floor_abs=FLOOR_U)
        if tag + ".p_rgh" in D.step:
            for p in range(min(2, len(D.step[tag + ".p_rgh"][1]))):
                v = probe_series(D, tag, p, "p_rgh")
                if v is not None:
                    put("%s %d p_rgh [Pa]" % (tag, p), v)
    for key, bad in sorted(D.unset.items()):
        P("  REJECTED (probe not located, written as -1e300): %s columns %s" % (key, bad))
    w = D.dt[sl]
    if len(w) and np.ptp(w) / np.median(w) < 1e-6:
        P("  %-40s fixed (spread %.1e of the median): not a series, not judged" % ("dt [s]", np.ptp(w) / np.median(w)))
    else:
        put("dt [s]", D.dt, "dt")
    if not film_probes(D) and "fluidProbes.U" in D.s05:
        tp, cols = D.s05["fluidProbes.U"]
        idx = np.searchsorted(D.t, tp - 1e-9)
        P("  0.5 s probe samples (legacy): sign flip between samples against odd step count between them")
        for p, c in enumerate(cols[:3]):
            if c is None:
                continue
            comp = int(np.argmax(np.mean(np.abs(c), axis=0)))
            s = np.sign(c[:, comp])
            flips = s[1:] * s[:-1] < 0
            odd = (idx[1:] - idx[:-1]) % 2 == 1
            P("      probe %d U%s: median |U| %.4g m/s; flip == odd-step-count in %d of %d sample pairs"
              % (p, "xyz"[comp], np.median(np.abs(c[:, comp])), int(np.sum(flips == odd)), len(flips)))
            res["probe%d parity" % p] = (int(np.sum(flips == odd)), int(len(flips)))
    OUT["alternation"] = {k: (v if not isinstance(v, dict) else {a: b for a, b in v.items()}) for k, v in res.items()}
    return res


def sec_growth(D, a, b, first=3):
    head("4. Growth of the alternating mode")
    P("  r[n] = d2[n+1]/d2[n], d2 = second difference of the per-step series; a 2-dt mode of gain g gives r = -g;")
    P("  a smooth trend on the x1.2 dt ramp gives r = +1.2.  Growth phase = a run of consecutive r < %.1f with" % GROW_R)
    P("  >= %d ratios and |d2| rising >= %.0f-fold (or >= %d ratios and >= %.0e-fold), ending at the largest |d2| so far."
      % (GROW_N, GROW_RISE, GROW_N_FAST, GROW_RISE_FAST))
    P("  Searched from step %d.  Run verdict: growth needs coincident phases in at least two series." % first)
    res = {}
    # Far-wall heat-flux extremes: zero until the pressure mode moves the wall-cell
    # temperature through dpdt, so they carry no trend.  In the three committed runs
    # the onset shows in the lid minimum and the floor maximum on every mesh (and in
    # nothing else recorded per step: Q_int is hidden under its own curvature until
    # the first dt cut).  Both extremes of both walls are examined.
    series = []
    # Pressure is the unstable variable itself, and its probes need no coupling to
    # show it.  (The heat-flux series see the mode only through the dpdt term: in a
    # 'dpdt no' run they stay flat while every pressure probe shows the onset.)
    # The first of these, the probe above the cell, gives the gain g that is
    # compared between pilots.
    for tag, k, where in GROWTH_P_PROBES:
        v = probe_series(D, tag, k, "p_rgh")
        if v is not None:
            series.append(("%s %d p_rgh" % (tag, k), v, dict()))
    # wall-normal velocity of a film probe: the cells next to the mid-height plane first (M7)
    for tag, k in (("fluidProbesExtra", 7), ("fluidProbesExtra", 8), ("fluidProbes", 0)):
        u = probe_series(D, tag, k, "U")
        if u is not None:
            series.append(("%s %d Ux" % (tag, k), u[:, 0], dict(floor_abs=FLOOR_U)))
            break
    for patch in ("bathLid", "bathFloor"):
        if patch in D.patch:
            for col, cn in ((0, "min"), (1, "max")):
                series.append(("%s %s flux" % (patch, cn), D.patch[patch][:, col], dict(floor_rel=0.0, floor_abs=5.0 * FLOOR_T)))
    series.append(("Q_int", D.patch[INTERFACE][:, 2], dict()))
    for label, v, kw in series:
        ok = np.isfinite(v)
        nfin = int(np.argmin(ok)) if not ok.all() else len(v)     # stop at the first missing value
        v = v[:nfin]
        if nfin < 12 or nfin < first + 6:
            P("  %-22s too few steps" % label)
            continue
        grow, decay, tail, near = growth_phases(v, first=first)
        st = alt_stats(v[len(v) - max(8, len(v) // 3):], **kw)
        vd = verdict_growth(grow, decay, tail, below=bool(st and st.get("below")))
        P("  %-22s -> %s" % (label, vd))
        if grow and not grow.get("slow"):
            P("      growth phase: steps %d..%d (%d ratios), gain g = median |r| %.3f (geometric mean %.3f); |d2| %.3e -> %.3e"
              " (x %.3g); %d such phase(s) in the run"
              % (grow["steps"][0], grow["steps"][1], grow["n"], grow["median"], grow["gmean"], grow["amp0"], grow["amp1"],
                 grow["rise"], grow["count"]))
            P("          ratios: " + " ".join("%+.2f" % x for x in grow["ratios"][:20]))
        if near["count"]:
            P("      not counted: %d run(s) of >= 4 ratios below %.1f without the required rise or not ending at a new"
              " maximum (longest %d)" % (near["count"], GROW_R, near["longest"]))
        if decay:
            P("      decay phase : steps %d..%d (%d ratios), geometric mean |r| %.3f; |d2| %.3e -> %.3e"
              % (decay["steps"][0], decay["steps"][1], decay["n"], decay["gmean"], decay["amp0"], decay["amp1"]))
        P("      last third of the run: r < 0 on %.1f %% of steps, geometric mean |r| %.3f%s"
          % (100 * tail["neg"], tail["gmean"], "; alternating part below the floor" if st and st.get("below") else ""))
        g = growth(v, a, b)
        if g is not None:
            P("      fixed window steps %d..%d: geometric mean |r| %.3f, median r %+.3f, r < 0 on %.1f %%"
              % (a, b, g["gmean"], g["median"], 100 * g["neg"]))
        gr_ramp = growth(v, *ASRUN_GAIN_STEPS) if first == 3 else None
        if grow is None and vd.startswith("SLOW GROWTH"):
            # counted as a phase over the last third, so that the run verdict can combine it
            grow = dict(steps=tail["steps"], n=tail["n"], median=tail["gain"], gmean=tail["gain"], slow=True,
                        amp0=float("nan"), amp1=float("nan"), rise=tail["gain"] ** tail["n"], count=1, ratios=[])
        res[label] = dict(verdict=vd, phase=grow, gain=grow["median"] if grow else None,
                          ramp_gain=(gr_ramp["gmean"] if gr_ramp and gr_ramp["neg"] == 1.0 else None),
                          gmean=grow["gmean"] if grow else (decay["gmean"] if decay else tail["gmean"]),
                          growth_steps=list(grow["steps"]) if grow else None,
                          growth_amp=[grow["amp0"], grow["amp1"]] if grow else None,
                          decay_gmean=decay["gmean"] if decay else None, tail_neg=tail["neg"], tail_gmean=tail["gmean"],
                          near=near)
    if res:
        vd, why = combined_growth(res)
        P("  RUN VERDICT: %s  (%s)" % (vd, why))
        OUT["growth_verdict"] = vd
    OUT["growth"] = {k: {a_: b_ for a_, b_ in v.items() if a_ != "phase"} for k, v in res.items()}
    return res


def sec_budget(D, T0, meta=None):
    head("5. Fluid energy budget (quarter domain)")
    ex = (meta or {}).get("expect", {})
    q = D.patch
    qi = q[INTERFACE][:, 2]
    qw = sum(v[:, 2] for k, v in q.items() if k != INTERFACE)
    dt = D.dt
    res = {}
    if D.pilot:
        H, M = on_steps(D, "fluidRhoH"), on_steps(D, "fluidMass")
        DP, KE2, MU = on_steps(D, "fluidDpdt"), on_steps(D, "fluidRhoMagSqrU"), on_steps(D, "fluidRhoU")
        miss = [k for k, v in (("int rho h", H), ("mass", M), ("int dpdt", DP), ("int rho|U|^2", KE2), ("int rho U", MU)) if v is None]
        if H is None or M is None:
            P("  per-step integrals missing (%s): budget not available" % ", ".join(miss))
            return res
        if miss:
            P("  WARNING: instruments missing, treated as zero: " + ", ".join(miss))
        z = np.zeros(D.n)
        DP = z if DP is None else DP
        KE = z if KE2 is None else 0.5 * KE2
        Gw = z if MU is None else -G * MU[:, 2]
        P("  instrument self-check: mass-weighted mean h at step 1 = %.6f J/kg (hs(T0) = %.6f); mass at step 1 %.9f kg"
          % (H[0] / M[0], hs(T0), M[0]) + ("; rho0*V = %.9f kg" % (rho_f(T0) * D.vol_fluid) if D.vol_fluid else ""))
        s = slice(1, None)                                   # steps 2..N
        dH = H[-1] - H[0]
        e_int, e_wall = float(np.sum(qi[s] * dt[s])), float(np.sum(qw[s] * dt[s]))
        # With one outer pass the energy equation of step n uses the dpdt left by step
        # n-1 (pEqn.H:119-122, EEqn.H:16).  With several passes the final energy solve
        # uses the dpdt of the previous pass of the same step, for which the sampled
        # end-of-step value is the closer proxy.
        lagged = ex.get("nOuter", 1) == 1
        e_dp_lag = float(np.sum(DP[:-1] * dt[1:]))
        e_dp_now = float(np.sum(DP[1:] * dt[1:]))
        e_dp = e_dp_lag if lagged else e_dp_now
        DPs = DP[:-1] if lagged else DP[1:]
        e_g = float(np.sum(Gw[s] * dt[s]))
        dKE = float(KE[-1] - KE[0])
        dM = float(M[-1] - M[0])
        # Mass leaves through the pressure reference cell, carrying that cell's enthalpy.
        href = np.full(D.n, H[0] / M[0])
        src = "h at step 1"
        tref = probe_series(D, "fluidProbesExtra", 0, "T")
        if tref is not None and np.isfinite(tref).all():
            href = hs(tref)
            src = "h(T) of the probe in the reference cell"
        e_sink = float(np.sum(href[1:] * (M[1:] - M[:-1])))
        R = dH - (e_int + e_wall + e_dp + e_g - dKE)
        Ropen = R - e_sink
        P("  window: steps 2..%d, t = %.6g .. %.6g s" % (D.n, D.t[0], D.t[-1]))
        P("  change of int(rho h) dV                      %+14.6e J" % dH)
        P("  interface heat, sum Q_int dt                 %+14.6e J" % e_int)
        P("  wall heat, sum Q_walls dt                    %+14.6e J" % e_wall)
        P("  pressure work, sum int(dpdt) dt              %+14.6e J   (%s; lagged %+.6e, unlagged %+.6e J)"
          % (e_dp, "dpdt of the previous step, as the equation uses it" if lagged else "end-of-step dpdt", e_dp_lag, e_dp_now))
        P("  gravity work, sum g.int(rho U) dt            %+14.6e J   (end-of-step U; the equation uses the predictor U)" % e_g)
        P("  kinetic energy change, -d int(rho K)         %+14.6e J   (end-of-step K; the equation uses the predictor K)" % (-dKE))
        P("  mass change                                  %+14.6e kg  (%.3e of the mass); enthalpy carried, sum h_ref dM = %+.6e J (%s)"
          % (dM, dM / M[0], e_sink, src))
        P("  RESIDUAL, rho*h basis                        %+14.6e J = %+.3f %% of interface energy" % (R, 100 * R / e_int if e_int else float("nan")))
        P("  RESIDUAL, open-system basis (sink at h_ref)  %+14.6e J = %+.3f %% of interface energy   [acceptance: |.| <= 1 %%]"
          % (Ropen, 100 * Ropen / e_int if e_int else float("nan")))
        rn = (H[1:] - H[:-1]) - dt[1:] * (qi[1:] + qw[1:] + DPs + Gw[1:]) + (KE[1:] - KE[:-1]) - href[1:] * (M[1:] - M[:-1])
        half = len(rn) // 2
        P("  per-step residual (open-system): median %+.3e J  = %+.3e J/kg per step; first half %+.3e, second half %+.3e J/step"
          % (np.median(rn), np.median(rn) / M[0], np.median(rn[:half]), np.median(rn[half:])))
        P("  bound from the limiter: 0.5*0.25^2 = 0.03125 J/kg per step")
        res = dict(dH=float(dH), e_int=e_int, e_wall=e_wall, e_dpdt_lag=e_dp_lag, e_dpdt_now=e_dp_now, e_g=e_g,
                   dKE=dKE, dM=dM, e_sink=e_sink, residual_rhoh=float(R), residual_open=float(Ropen),
                   residual_open_pct=float(100 * Ropen / e_int) if e_int else None,
                   per_step_J_per_kg=float(np.median(rn) / M[0]))
        Hs = on_steps(D, "cellH")
        if Hs is not None and D.vol_solid:
            P_src = Q_SOLID * D.vol_solid if ex.get("heatSource", True) else 0.0
            e_solid = P_src * (D.t[-1] - D.t[0]) - RHO_SOLID * (Hs[-1] - Hs[0])
            P("  solid side: source %.6f W (if on); source*t - d(2700 int h) = %+.6e J against fluid-side interface heat %+.6e J"
              " (difference %+.3f %%)" % (P_src, e_solid, e_int, 100 * (e_solid - e_int) / e_int if e_int else float("nan")))
            res["e_solid_released"] = float(e_solid)
    else:
        if "bathTemperatureMean" not in D.s05 or not D.vol_fluid:
            P("  no bath-mean temperature samples: budget not available")
            return res
        ts, Tm = D.s05["bathTemperatureMean"]
        M0 = rho_f(T0) * D.vol_fluid
        cum_i = np.concatenate([[0.0], np.cumsum(qi * dt)])
        cum_w = np.concatenate([[0.0], np.cumsum(qw * dt)])
        tt = np.concatenate([[0.0], D.t])
        ei, ew = np.interp(ts, tt, cum_i), np.interp(ts, tt, cum_w)
        Eb = M0 * (hs(Tm) - hs(T0))
        P("  LEGACY basis: stored = rho0*V*[hs(Tmean) - hs(T0)], fixed mass %.5f kg, 0.5 s samples" % M0)
        P("  t [s]     interface in [J]   walls [J]   bath stored [J]   leak [J]   leak/interface")
        done = set()
        for T in (2.0, 10.0, 30.0, 100.0, ts[-1]):
            if T > ts[-1] * (1 + 1e-9):
                continue
            k = int(np.argmin(np.abs(ts - T)))
            if k in done:
                continue
            done.add(k)
            leak = ei[k] + ew[k] - Eb[k]
            P("  %-8g %14.4f %12.4f %15.4f %12.4f %10.2f %%" % (ts[k], ei[k], ew[k], Eb[k], leak, 100 * leak / ei[k] if ei[k] else float("nan")))
        k = len(ts) - 1
        res = dict(e_int=float(ei[k]), e_wall=float(ew[k]), stored=float(Eb[k]), leak=float(ei[k] + ew[k] - Eb[k]), t=float(ts[k]))
        P("  bath-mean temperature: minimum %.6f K = T0 %+.3e K at t = %g s" % (Tm.min(), Tm.min() - T0, ts[int(np.argmin(Tm))]))
    OUT["budget"] = res
    return res


def sec_extrema(D, L, T0):
    head("6. Extrema")
    res = {}
    k0 = T_FROM_STEP - 1

    def tmin(v, what, key, drop, src):
        """Minimum over the whole run (information) and from step T_FROM_STEP on (judged)."""
        if not np.isfinite(v).any():
            return
        k = int(np.nanargmin(v))
        P("  %s: minimum %.9f K = T0 %+.3e K at step %d (t = %.6g s)" % (what, v[k], v[k] - T0, k + 1, D.t[min(k, D.n - 1)]))
        res[key + "_all"] = float(v[k] - T0)
        if len(v) > k0 and np.isfinite(v[k0:]).any():
            k = k0 + int(np.nanargmin(v[k0:]))
            P("      from step %d on (start-up ringing excluded): T0 %+.3e K at step %d   [acceptance, %s: >= T0 - %g K]"
              % (T_FROM_STEP, v[k] - T0, k + 1, src, drop))
            res[key] = float(v[k] - T0)

    v = on_steps(D, "fluidTmin")
    if v is not None:
        tmin(v, "local minimum fluid temperature", "Tmin_minus_T0", TMIN_DROP_MAX, "R2-10")
    elif L is not None and np.isfinite(L["tmin"]).any():
        tmin(L["tmin"], "local minimum fluid temperature (log 'Min/max T', after the energy solve)",
             "Tmin_minus_T0", TMIN_DROP_MAX, "R2-10")
    v = on_steps(D, "bathTemperatureMean")
    if v is not None:
        tmin(v, "bath mean temperature", "Tmean_min_minus_T0", TMEAN_DROP_MAX, "N5")
        P("      end of run: %+.6e K above T0" % (v[np.isfinite(v)][-1] - T0))
    elif "bathTemperatureMean" in D.s05:
        ts, Tm = D.s05["bathTemperatureMean"]
        k = int(np.argmin(Tm))
        P("  bath mean temperature (0.5 s samples): minimum %.6f K = T0 %+.3e K at t = %g s   [acceptance, N5: >= T0 - %g K]"
          % (Tm[k], Tm[k] - T0, ts[k], TMEAN_DROP_MAX))
        res["Tmean_min_minus_T0"] = float(Tm[k] - T0)
    v = on_steps(D, "plumeSpeed")
    if v is not None:
        k = int(np.nanargmax(v))
        P("  max|U|: peak %.6g m/s at step %d (t = %.6g s); median %.6g; last %.6g m/s" % (v[k], k + 1, D.t[k], np.nanmedian(v), v[-1]))
        res["Umax_peak"], res["Umax_last"], res["Umax_median"] = float(v[k]), float(v[-1]), float(np.nanmedian(v))
    elif "plumeSpeed" in D.s05:
        ts, u = D.s05["plumeSpeed"]
        P("  max|U| (0.5 s samples): first %.6g at t = %g s; median %.6g; max %.6g; last %.6g m/s" % (u[0], ts[0], np.median(u), u.max(), u[-1]))
        res["Umax_peak"], res["Umax_median"] = float(u.max()), float(np.median(u))
    for name, rows in D.minmax.items():
        if not rows:
            continue
        a = np.array([[r[1], r[2]] for r in rows])
        k = int(np.argmax(np.abs(a[:, 1])))
        P("  %-10s min %.9g  max %.9g over the run; first step: min %.9g max %.9g; location of the largest max: (%s)"
          % (name, a[:, 0].min(), a[:, 1].max(), a[0, 0], a[0, 1], rows[k][4]))
        res["minmax_" + name] = [float(a[:, 0].min()), float(a[:, 1].max())]
    OUT["extrema"] = res
    return res


def sec_asrun(D, ref_post):
    head("7. Comparison with the committed as-run monitors")
    R = load(ref_post)
    P("  reference: %s (%d steps, t_end %.6g s)" % (ref_post, R.n, R.t[-1]))
    checks = []                                   # (what, value, same?, tolerance)

    def chk(what, val, ok, tol):
        checks.append((what, val, bool(ok), tol))
        P("  %-52s %-26s -> %s   [%s]" % (what, val, "same" if ok else "DIFFERS", tol))

    n = min(D.n, R.n)
    sig6 = lambda x: np.array([float("%.6g" % v) for v in x])
    same = sig6(D.t[:n]) == sig6(R.t[:n])
    nsame = int(np.argmin(same)) if not same.all() else n
    P("  step times equal to 6 significant digits for steps 1..%d" % nsame)
    fa, fb = first_cut(D.dt), first_cut(R.dt)
    # The ramp is pure arithmetic (x1.2 per step, then the write-time rounding of
    # Time::adjustDeltaT, which can move a step by dt/(2*time to next write)); a
    # last-bit difference between machines can flip that rounding, so the test is
    # a tolerance, not equality.  It stops two steps before the committed cut:
    # the cut itself may come one step earlier (see below).
    nr = max(1, min((fb - 2) if fb else n, n))
    rel = np.abs(D.t[:nr] / R.t[:nr] - 1.0)
    chk("step times 1..%d (the x1.2 ramp)" % nr, "max rel. difference %.2e" % rel.max(), rel.max() <= 2e-3,
        "<= 2e-3; 6 equal digits expected")
    # The step of the first cut is where the Courant number first exceeds maxCo.
    # It depends on the amplitude of the growing mode, which starts at the level
    # of the linear-solver tolerance and so differs between 4 and 16 ranks; at a
    # gain of 8-10 per step, one step either way covers a factor of 8-10 in it.
    if fb:
        chk("first dt cut (committed: step %d)" % fb, "step %s" % fa, fa is not None and abs(fa - fb) <= 1, "step %d +-1" % fb)
    if fa and fb:
        P("      dt after the first cut: %.4g s (committed %.4g s, ratio %.2f).  Not a test: the cut is dt*maxCo/Co"
          % (D.dt[fa - 1], R.dt[fb - 1], D.dt[fa - 1] / R.dt[fb - 1]))
        P("      (setMultiRegionDeltaT.H:48-58) and Co at the cut lies anywhere between 3 and about 3 x the gain per step")
        P("      (committed coarse run: Co 12.4; archived medium log: Co 4.2), i.e. the dt after it between 0.1 and 1 x the dt before.")
    res = dict(steps_identical=nsame, first_cut=(fa, fb), ramp_max_rel=float(rel.max()))
    for T in (0.5, 1.0, 2.0):
        if D.t[-1] >= T * (1 - 1e-9) and R.t[-1] >= T * (1 - 1e-9):
            a, b = int(np.searchsorted(D.t, T * (1 - 1e-9)) + 1), int(np.searchsorted(R.t, T * (1 - 1e-9)) + 1)
            if T == 2.0:
                chk("steps to t = 2 s (committed: %d)" % b, "%d (%+.1f %%)" % (a, 100.0 * (a - b) / b), abs(a - b) <= 0.05 * b, "+-5 %")
            else:
                P("  steps to t = %g s: %d (reference %d, %+.1f %%)" % (T, a, b, 100.0 * (a - b) / b))
            res["steps_to_%g" % T] = (a, b)
    lo, hi = 0.5, min(2.0, D.t[-1], R.t[-1])
    ma, mb = (D.t > lo) & (D.t <= hi), (R.t > lo) & (R.t <= hi)
    if ma.sum() > 3 and mb.sum() > 3:
        P("  median dt over %g-%g s: %.4g s (committed %.4g s, %+.1f %%)"
          % (lo, hi, np.median(D.dt[ma]), np.median(R.dt[mb]), 100 * (np.median(D.dt[ma]) / np.median(R.dt[mb]) - 1)))
    # film probe: N4 'probe-0 |Ux| about 0.026 m/s flipping every step'
    fp = film_probes(D)
    if fp and "fluidProbes.U" in R.s05 and R.s05["fluidProbes.U"][1][0] is not None and fa:
        label, what, u = fp[0]
        k1 = min(fa + 9, D.n - 2)
        tr, cr = R.s05["fluidProbes.U"]
        mr = tr <= D.t[-1] * (1 + 1e-9)
        if mr.any():
            ref = float(np.median(np.abs(cr[0][mr, 0])))
            mine = float(np.nanmedian(np.abs(u[k1:, 0])))
            par = sign_changes(u[k1:, 0])
            chk("%s median |Ux| from step %d" % (label, k1 + 1), "%.4g m/s (committed %.4g)" % (mine, ref),
                abs(mine / ref - 1.0) <= 0.25, "within 25 % of the committed 0.5 s samples of probe 0")
            chk("%s Ux sign changes" % label, "%.1f %% of steps" % (100 * par), par >= 0.95, ">= 95 %; committed: every step")
            res["probe_medabs"], res["probe_parity"] = mine, par
    m = min(nsame, n)
    if m > 3:
        qa, qb = D.patch[INTERFACE][:m, 2], R.patch[INTERFACE][:m, 2]
        rel = np.abs(qa[2:] - qb[2:]) / np.maximum(np.abs(qb[2:]), 1e-300)
        P("  interface heat rate over those steps: max relative difference %.3e (steps 3..%d)" % (rel.max(), m))
        res["qint_maxrel"] = float(rel.max())
    P("  0.5 s samples (this run: per-step series taken at the reference sample times)")
    for key in ("cellTemperatureMax", "cellTemperatureMean", "bathTemperatureMean", "bathTemperatureMax",
                "plumeSpeed", "canSurfaceTemperature"):
        if key not in R.s05:
            continue
        tr, vr = R.s05[key]
        if key in D.s05:
            tm, vm = D.s05[key]
        elif key in D.step:
            tm, vm = D.step[key]
        else:
            continue
        rows = []
        for tt, v in zip(tr, vr):
            if tt > D.t[-1] * (1 + 1e-9):
                break
            k = int(np.argmin(np.abs(tm - tt)))
            if abs(tm[k] - tt) <= 1e-6 * max(1.0, tt):
                rows.append("t=%g: %.8g vs %.8g" % (tt, vm[k], v))
        P("      %-24s %s" % (key, ";  ".join(rows) if rows else "no common sample times"))
    nd = sum(1 for c in checks if not c[2])
    P("  AS-RUN REPRODUCTION: %s" % ("all %d checks the same as the committed run" % len(checks) if not nd
                                      else "%d of %d checks DIFFER from the committed run" % (nd, len(checks))))
    res["checks"] = [dict(what=c[0], value=c[1], same=c[2], tolerance=c[3]) for c in checks]
    OUT["asrun"] = res
    return res


def summary(alt, gr, bud, ext, meta, asrun=None, info=None, n_steps=None):
    head("8. Acceptance summary (N4, N5, R2-10; the numbers are those of PILOTS.md, section C)")
    fails, judged, tested = [], 0, [0]

    def line(label, val, ok, thr):
        P("  %-60s %-15s %-5s (%s)" % (label, val, "PASS" if ok else "FAIL", thr))
        tested[0] += 1
        if not ok:
            fails.append(label)

    # a run that stopped early cannot pass, whatever its statistics up to that point
    if info is not None and (info["fatal"] or not info["end"]):
        line("run reached End", "no", False, "log: %s" % (info["fatal"][0][:60] if info["fatal"] else "no 'End' line (stopped, or still running)"))
    want = (meta or {}).get("steps")
    if want and n_steps is not None and n_steps < 0.98 * want:
        line("time steps recorded", "%d of %d" % (n_steps, want), False, "the run is incomplete")

    film = 0
    for k, s in alt.items():
        if not isinstance(s, dict) or not s.get("kind"):
            continue
        judged += 1
        film += s["kind"] == "film"
        if s.get("const") or s.get("below"):
            P("  %-60s %-15s %-5s (alternating part %.1e under the floor %.1e; rho1 %+.2f%s not judged)"
              % (k, "below floor", "-", s["amp"], s["floor"], s["rho1"],
                 ", sign changes %.0f %%" % (100 * s["parity"]) if s["kind"] == "film" else ""))
            continue
        line("rho1 " + k, "%+.3f" % s["rho1"], s["rho1"] > RHO1_MIN, "> %g" % RHO1_MIN)
        if s["kind"] == "Q":
            line("alt/mean " + k, "%.3e" % s["alt"], s["alt"] <= ALT_Q_MAX, "<= %g; as run 1.5e-3" % ALT_Q_MAX)
        elif s["kind"] == "film":
            line("alternating amplitude " + k, "%.3e m/s" % s["amp"], s["amp"] <= AMP_U_MAX,
                 "<= %.1e m/s = 1 %% of P0's %.3f" % (AMP_U_MAX, P0_PROBE_AMP))
            line("sign changes " + k, "%.1f %%" % (100 * s["parity"]), s["parity"] <= PARITY_MAX,
                 "<= %.0f %% of steps" % (100 * PARITY_MAX))
    if not film:
        P("  (no per-step film probe in these data: the probe criteria are not evaluated)")
    if bud.get("residual_open_pct") is not None:
        line("energy residual / interface energy", "%+.3f %%" % bud["residual_open_pct"], abs(bud["residual_open_pct"]) <= 1.0, "|.| <= 1 %")
    if "Tmean_min_minus_T0" in ext:
        line("bath MEAN T - T0, minimum from step %d" % T_FROM_STEP, "%+.3e K" % ext["Tmean_min_minus_T0"],
             ext["Tmean_min_minus_T0"] >= -TMEAN_DROP_MAX, ">= -%g K, N5" % TMEAN_DROP_MAX)
    if "Tmin_minus_T0" in ext:
        line("LOCAL minimum fluid T - T0, from step %d" % T_FROM_STEP, "%+.3e K" % ext["Tmin_minus_T0"],
             ext["Tmin_minus_T0"] >= -TMIN_DROP_MAX, ">= -%g K, R2-10" % TMIN_DROP_MAX)
    if fails:
        verdict = "FAIL on %d line(s)" % len(fails)
    elif not tested[0] and not judged:
        verdict = "NOT EVALUATED (no judged quantity in these data)"
    else:
        verdict = "PASS, no line failed (%d tested, %d series judged)" % (tested[0], judged)
    P("  ACCEPTANCE (candidate settings): %s" % verdict)
    OUT["acceptance"] = dict(passed=bool(not fails and (tested[0] or judged)), failed=fails, judged_series=judged)
    for k, g in gr.items():
        P("  alternating mode, %-22s %s%s" % (k, g["verdict"], " (gain g %.3g over steps %d..%d)"
                                               % (g["gain"], g["growth_steps"][0], g["growth_steps"][1]) if g.get("phase") else ""))
    if gr:
        P("  alternating mode, run verdict (section B): %s" % combined_growth(gr)[0])
        ref = "%s %d p_rgh" % GROWTH_P_PROBES[0][:2]
        if gr.get(ref, {}).get("ramp_gain"):
            P("  gain g to compare between as-run pilots (%s, %s): %.3g = geometric mean of -d2[n+1]/d2[n] over"
              " steps %d-%d of the ramp" % (ref, GROWTH_P_PROBES[0][2], gr[ref]["ramp_gain"], ASRUN_GAIN_STEPS[0], ASRUN_GAIN_STEPS[1]))
        elif gr.get(ref, {}).get("phase"):
            P("  gain g to compare between pilots (%s, %s): %.3g, phase from step %d"
              % (ref, GROWTH_P_PROBES[0][2], gr[ref]["gain"], gr[ref]["growth_steps"][0]))
    if asrun and asrun.get("checks"):
        nd = sum(1 for c in asrun["checks"] if not c["same"])
        P("  as-run reproduction (section A): %d of %d checks the same as the committed run" % (len(asrun["checks"]) - nd, len(asrun["checks"])))
    if meta:
        P("  prediction for %s: %s" % (meta["id"], meta["prediction"]))
        P("  decided by: %s" % meta["observable"])
        P("  threshold : %s" % meta["threshold"])


# ----------------------------------------------------------------------------
def selftest():
    """Synthetic data in the harness formats: the budget must close to round-off
    and a planted 2-dt mode must be recovered.  Tests arithmetic and parsers,
    NOT the real instruments."""
    import shutil
    import tempfile
    # scratch files next to this script (not in the system temp directory); removed at the end
    d = tempfile.mkdtemp(prefix=".pm_selftest_", dir=os.path.dirname(os.path.abspath(__file__)))
    post = os.path.join(d, "postProcessing")
    n, M0, V = 400, 0.1, 1.0e-4
    rng = np.random.default_rng(1)
    dt = np.minimum(1e-4 * 1.2 ** np.arange(1, n + 1), 8.5e-3) * (1 + 0.02 * (-1.0) ** np.arange(n))
    t = np.cumsum(dt)
    qi = 0.3 * np.sqrt(t) * (1 + 0.05 * (-1.0) ** np.arange(n))
    qw = -0.01 * t
    DP = 0.02 * (-1.0) ** np.arange(n) + 0.001 * rng.standard_normal(n)
    MUz = 1e-5 * np.sin(5 * t)
    KE2 = 2e-6 * (1 + np.cos(3 * t))
    M = M0 * (1 - 1e-5 * t)
    href = hs(T0_DEFAULT)
    H = np.zeros(n)
    H[0] = M[0] * href
    planted = 3.0e-6                                   # J per step, a deliberate sink
    for k in range(1, n):
        H[k] = (H[k - 1] + dt[k] * (qi[k] + qw[k] + DP[k - 1] - G * MUz[k]) - 0.5 * (KE2[k] - KE2[k - 1])
                + href * (M[k] - M[k - 1]) - planted)

    def wr(name, fname, hdr, rows, region="fluid"):
        p = os.path.join(post, region, name, "0")
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, fname), "w") as f:
            f.write(hdr)
            f.write("\n".join(rows) + "\n")
    wr("wallHeatFlux", "wallHeatFlux.dat", "# Wall heat-flux\n# Time\tpatch\tmin\tmax\tintegral\n",
       ["%.10g\tbathLid\t%.15e\t0\t%.15e\n%.10g\tfluid_to_battery\t0\t0\t%.15e"
        % (t[k], qw[k] + (1e-9 * (-2.0) ** min(k - 30, 30) if k >= 30 else 0.0), qw[k], t[k], qi[k]) for k in range(n)])
    h4 = "# Region        : all fluid\n# Cells         : 10\n# Volume        : %.15e\n# Time\tX\n" % V
    sc = lambda v: ["%.10g\t%.15e" % (t[k], v[k]) for k in range(n)]
    wr("step_fluidRhoH", "volFieldValue.dat", h4, sc(H))
    wr("step_fluidMass", "volFieldValue.dat", h4, sc(M))
    wr("step_fluidDpdt", "volFieldValue.dat", h4, sc(DP))
    wr("step_fluidRhoMagSqrU", "volFieldValue.dat", h4, sc(KE2))
    wr("step_fluidRhoU", "volFieldValue.dat", h4, ["%.10g\t(0 0 %.15e)" % (t[k], MUz[k]) for k in range(n)])
    wr("step_fluidTmin", "volFieldValue.dat", h4, sc(np.full(n, T0_DEFAULT - 2e-6)))
    wr("step_bathTemperatureMean", "volFieldValue.dat", h4, sc(T0_DEFAULT + 1e-3 * t))
    D = load(post)
    assert D.pilot and D.n == n
    b = sec_budget(D, T0_DEFAULT)
    want = -planted * (n - 1)
    assert abs(b["residual_open"] - want) < 1e-9, (b["residual_open"], want)
    assert abs(b["e_dpdt_lag"] - np.sum(DP[:-1] * dt[1:])) < 1e-9   # times are printed to 10 digits
    a = alt_stats(qi[50:])
    assert a["rho1"] < -0.9 and a["zig"] > 0.95 and not a["below"] and abs(a["alt"] - 0.05) < 0.01, a
    sm = alt_stats(np.sqrt(t[50:]))                         # sampled with a dt that alternates by 2 %
    assert sm["rho1"] > 0.9 and sm["alt"] < 1e-3, sm
    sm = alt_stats(0.3 * np.sqrt(3.2 + 0.2 * np.arange(85)))  # smooth heat rate at a fixed dt of 0.2 s
    assert sm["rho1"] > 0.9 and sm["alt"] < 1e-6 and sm["alt"] < ALT_Q_MAX, sm
    # floors: noise is reported as below floor instead of failing rho1 > -0.2
    nz = alt_stats(1e-6 + 1e-12 * rng.standard_normal(600), floor_abs=FLOOR_U)
    assert nz["rho1"] < -0.3 and nz["below"], nz
    # a pure 2-dt mode of amplitude A has alternating part A (third difference / 8)
    pure = alt_stats(0.026 * (-1.0) ** np.arange(300), floor_abs=FLOOR_U)
    assert abs(pure["amp"] - 0.026) < 1e-12 and not pure["below"] and pure["rho1"] < -0.99, pure
    # growth rule: the planted mode (gain 2 from step 30) is found, noise and a smooth trend are not
    grow, decay, tail, near = growth_phases(D.patch["bathLid"][:, 0])
    assert grow and abs(grow["gmean"] - 2.0) < 0.2 and 30 <= grow["steps"][0] <= 45 and grow["rise"] >= GROW_RISE, grow
    g = growth(D.patch["bathLid"][:, 0], 46, 58)
    assert g and abs(g["gmean"] - 2.0) < 0.05 and g["neg"] == 1.0, g
    assert growth_phases(np.sqrt(t))[0] is None
    hits = sum(growth_phases(rng.standard_normal(600))[0] is not None for _ in range(200))
    assert hits <= 2, hits                                   # the bare '>= 4 ratios' rule gives about 165
    hits = sum(growth_phases((-0.6) ** np.arange(600) + 1e-9 * rng.standard_normal(600))[0] is not None for _ in range(200))
    assert hits <= 2, hits
    fast = np.concatenate([np.zeros(30), 1e-9 * (-40.0) ** np.arange(1, 8)])     # diverges within 7 steps
    gf = growth_phases(fast + 1e-12 * rng.standard_normal(37))[0]
    assert gf and gf["n"] < GROW_N and gf["rise"] >= GROW_RISE_FAST, gf
    # a gain of 1.05 per step never gives a ratio below -1.1: it is reported as slow growth
    k6 = np.arange(600)
    slow = np.sqrt(0.05 + 8.5e-3 * k6) + 1e-9 * (-1.05) ** k6 + 1e-11 * rng.standard_normal(600)
    gs, ds, tl, nr = growth_phases(slow, first=26)
    assert gs is None and verdict_growth(gs, ds, tl).startswith("SLOW GROWTH") and abs(tl["gain"] - 1.05) < 0.01, (gs, tl)
    sat = 0.02 * (-1.0) ** k6 * (1 + 0.05 * rng.standard_normal(600))            # saturated, not growing
    gs, ds, tl, nr = growth_phases(sat)
    assert verdict_growth(gs, ds, tl).startswith("SUSTAINED"), (gs, tl)
    # run verdict: two coincident phases are growth, one alone is not
    ph = lambda a_, b_: dict(verdict="GROWTH", phase=dict(steps=(a_, b_)))
    none = dict(verdict="decay", phase=None)
    assert combined_growth(dict(lid=ph(26, 40), probe=ph(27, 40), q=ph(33, 40)))[0] == "GROWTH"
    assert combined_growth(dict(lid=ph(26, 40), probe=none, q=none))[0].startswith("growth in ONE")
    assert combined_growth(dict(lid=ph(26, 40), probe=ph(300, 310), q=none))[0].startswith("growth in ONE")
    assert combined_growth(dict(lid=none, probe=none, q=none))[0] == "no growth; decay"
    # probes: a column holding the unset value (-1e300) is rejected, the others are kept
    um = 1e-3 * np.tanh(t)[:, None] * np.array([0.0, 0.0, 1.0]) + 0.02 * ((-1.0) ** np.arange(n))[:, None] * np.array([1.0, 0.0, 0.0])
    wr("step_fluidProbes", "U", "# Probe 0 (0.00915 0.0002 0.065)\n# Probe 1 (0.002 0.002 0.105)\n#   Time\n",
       ["%.10g (-1e+300 -1e+300 -1e+300) (%.15e %.15e %.15e)" % (t[k], um[k, 0], um[k, 1], um[k, 2]) for k in range(n)], region="")
    ex = ["(0 0 0)"] * 7
    wr("step_fluidProbesExtra", "U", "#   Time\n",
       ["%.10g %s (%.15e %.15e %.15e) (%.15e 0 %.15e)" % (t[k], " ".join(ex), um[k, 0], um[k, 1], um[k, 2], 1e-9 * rng.standard_normal(), um[k, 2])
        for k in range(n)], region="")
    D = load(post)
    assert D.unset == {"step_fluidProbes.U": [0]}, D.unset
    assert probe_series(D, "fluidProbes", 0) is None and probe_series(D, "fluidProbes", 1) is not None
    fp = film_probes(D)
    assert [f[0] for f in fp] == ["fluidProbesExtra 7", "fluidProbesExtra 8", "fluidProbesExtra 5", "fluidProbesExtra 6"], [f[0] for f in fp]
    alt = sec_alt(D, 60)
    f7 = alt["fluidProbesExtra 7 Ux [m/s]"]
    assert f7["kind"] == "film" and abs(f7["amp"] - 0.02) < 1e-6 and f7["parity"] == 1.0 and not f7["below"], f7
    f8 = [v for k, v in alt.items() if k.startswith("fluidProbesExtra 8")][0]
    assert f8["below"], f8                                   # smooth film flow plus 1e-9 noise
    assert "dt [s]" in alt and alt["dt [s]"]["kind"] == "dt"   # this synthetic dt alternates by 2 %
    # temperature thresholds: the mean is judged at 1e-5 K, the local minimum at 1e-3 K, both from step 13
    ext = sec_extrema(D, None, T0_DEFAULT)
    assert abs(ext["Tmin_minus_T0"] + 2e-6) < 1e-9 and ext["Tmean_min_minus_T0"] > 0, ext
    # log parser on a fabricated two-step log
    lg = os.path.join(d, "log")
    with open(lg, "w") as f:
        f.write("nProcs : 4\n    Source: limitU\n    State: inactive\ndeltaT = 0.00012\nTime = 0.00012\n\n"
                "Solving for fluid region fluid\nDILUPBiCGStab:  Solving for Ux, Initial residual = 1, Final residual = 1e-9, No Iterations 3\n"
                "Lookup equation relaxation factor for UFinal\nEquation relaxation factor for UFinal is 1 (from Function1)\n"
                "DILUPBiCGStab:  Solving for h, Initial residual = 1, Final residual = 1e-9, No Iterations 1\nMin/max T:308.15 308.16\n"
                "GAMG:  Solving for p_rgh, Initial residual = 0.5, Final residual = 1e-3, No Iterations 9\n"
                "GAMG:  Solving for p_rgh, Initial residual = 0.1, Final residual = 1e-8, No Iterations 20\n"
                "DILUPBiCGStab:  Solving for Ux, Initial residual = 0.25, Final residual = 1e-9, No Iterations 3\n"
                "DILUPBiCGStab:  Solving for h, Initial residual = 0.5, Final residual = 1e-9, No Iterations 1\n"
                "    Did not find location (0.1 0.1 0.1) in any cell. Skipping location.\n"
                "    Location (0.00915 0.0002 0.065) seems to be on multiple domains: cell 3 on my domain 1\n"
                "--> execute() function object 'step_fieldMinMax'\n"
                "time step continuity errors (fluid): sum local = 1e-12, global = 1e-13, cumulative = 1e-13\n\n"
                "Solving for solid region battery\nDICPCG:  Solving for h, Initial residual = 1, Final residual = 1e-11, No Iterations 3\n"
                "Min/max T:308.15 308.15\nExecutionTime = 0.3 s  ClockTime = 1 s\n\n"
                "Region: fluid Courant Number mean: 1e-6 max: 2e-5\nRegion: battery Diffusion Number mean: 0.01 max: 0.03\n"
                "deltaT = 0.000144\nTime = 0.000264\n\nSolving for fluid region fluid\n"
                "GAMG:  Solving for p_rgh, Initial residual = 0.4, Final residual = 1e-3, No Iterations 9\n"
                "ExecutionTime = 0.5 s  ClockTime = 2 s\nEnd\n")
    L, info = parse_log(lg)
    assert list(L["n_p"]) == [2, 1] and list(L["n_fluid"]) == [1, 1] and list(L["n_Ux"]) == [2, 0], L
    assert L["U_r0"][0] == 1.0 and L["U_rl"][0] == 0.25 and L["h_r0"][0] == 1.0 and L["h_rl"][0] == 0.5, (L["U_rl"], L["h_rl"])
    assert len(info["probe_not_found"]) == 1 and len(info["probe_multi"]) == 1 and len(info["fo_lines"]) == 1, info
    assert info["fvopt"] == {"limitU": "inactive"} and info["relax"] == {"UFinal=1": 1} and info["end"]
    assert abs(L["dt"][1] - 0.000144) < 1e-12 and abs(L["co_max"][1] - 2e-5) < 1e-12
    P()
    shutil.rmtree(d, ignore_errors=True)
    P("SELFTEST PASSED (synthetic files, removed afterwards): budget residual recovered to %.1e J, 2-dt statistics and "
      "floors, growth rule and run verdict, rejected probe column, temperature thresholds, log parser"
      % abs(b["residual_open"] - want))


def read_case_inputs(case):
    """Take the heat source and the coolant heat-capacity law from the case itself.

    The module constants are the baseline inputs.  Runs with another heat rate or a
    constant heat capacity carry their own values in system/battery/fvOptions and
    constant/fluid/thermophysicalProperties; using the constants for them gave a wrong
    solid-side balance (20 W run) and a wrong enthalpy for the expelled liquid (constant-cp twin).
    """
    global CP_C, Q_SOLID
    f = os.path.join(case, "system", "battery", "fvOptions")
    if os.path.isfile(f):
        m = re.search(r"^\s*h\s+\(\s*([0-9.eE+-]+)\s+[0-9.eE+-]+\s*\)\s*;", open(f).read(), re.M)
        if m:
            Q_SOLID = float(m.group(1))
    f = os.path.join(case, "constant", "fluid", "thermophysicalProperties")
    if os.path.isfile(f):
        m = re.search(r"CpCoeffs<8>\s+\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)", open(f).read())
        if m:
            CP_C = (float(m.group(1)), float(m.group(2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", help="pilot case directory (contains postProcessing/, log.chtMultiRegionFoam, pilot.json)")
    ap.add_argument("--post", help="a postProcessing directory (committed run)")
    ap.add_argument("--log", help="solver log (plain or .gz)")
    ap.add_argument("--asrun", help="committed postProcessing directory to compare a P0-type run against")
    ap.add_argument("--tmax", type=float, help="use only data up to this time [s]")
    ap.add_argument("--growth", help="step range a:b for the growth ratios (default: cap step .. +40, or 15:45)")
    ap.add_argument("--from-step", type=int, help="first step of the alternation statistics")
    ap.add_argument("--T0", type=float, default=T0_DEFAULT)
    ap.add_argument("--json", help="also write the numbers to this file")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    meta = None
    if a.case:
        read_case_inputs(a.case)
        post = os.path.join(a.case, "postProcessing")
        log = a.log or os.path.join(a.case, "log.chtMultiRegionFoam")
        pj = os.path.join(a.case, "pilot.json")
        if os.path.isfile(pj):
            meta = json.load(open(pj))
    elif a.post:
        post, log = a.post, a.log
    else:
        ap.error("give a case directory or --post")
    if not os.path.isdir(post):
        sys.exit("no postProcessing directory at %s (has the pilot been run?)" % post)
    D = load(post)
    if a.tmax:
        n = int(np.searchsorted(D.t, a.tmax * (1 + 1e-9)))
        D.t, D.dt, D.n = D.t[:n], D.dt[:n], n
        D.patch = {k: v[:n] for k, v in D.patch.items()}
        for st in (D.step, D.s05):
            for k in list(st):
                t, v = st[k]
                m = t <= a.tmax * (1 + 1e-9)
                st[k] = (t[m], [None if c is None else c[m] for c in v] if isinstance(v, list) else v[m])
    L = info = None
    if log and os.path.isfile(log):
        L, info = parse_log(log, max_steps=D.n if a.tmax else None)
    P("pilot_metrics: %s   mode: %s%s" % (a.case or a.post, "pilot (per-step instruments)" if D.pilot else "legacy (as-run monitors)",
                                           "   pilot %s: %s" % (meta["id"], meta["change"]) if meta else ""))
    OUT["mode"] = "pilot" if D.pilot else "legacy"
    sec_log(L, info, meta)
    if D.n < 8:
        P("\nonly %d time steps were recorded by the monitors: nothing further can be computed "
          "(see the end of the log)" % D.n)
        return
    sec_dt(D, L)
    ncap = (meta or {}).get("nCap")
    fc = first_cut(D.dt)
    n0 = a.from_step or (ncap + 10 if ncap else (fc + 10 if fc else 40))
    n0 = max(2, min(n0, max(2, D.n - 10)))
    alt = sec_alt(D, n0)
    if a.growth:
        ga, gb = [int(x) for x in a.growth.split(":")]
    elif ncap:
        ga, gb = ncap, min(ncap + 40, D.n - 2)
    elif fc:
        ga, gb = max(3, fc - 10), min(fc - 1, D.n - 2)
    else:
        ga, gb = 15, min(45, D.n - 2)
    # fixed-dt pilots: search for growth from the step after the cap is reached
    gr = sec_growth(D, ga, gb, first=(ncap + 1) if ncap else 3)
    bud = sec_budget(D, a.T0, meta)
    ext = sec_extrema(D, L, a.T0)
    asr = sec_asrun(D, a.asrun) if a.asrun else None
    summary(alt, gr, bud, ext, meta, asr, info, D.n)
    if a.json:
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        with open(a.json, "w") as f:
            json.dump(clean(OUT), f, indent=1)
        P("\nwrote " + a.json)


if __name__ == "__main__":
    main()
