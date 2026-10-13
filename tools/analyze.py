#!/usr/bin/env python3
"""
Post-process the 18650 immersion-cooling run.

Reads only postProcessing/ -- the small monitor time series -- so the results
table and plots can be rebuilt without the (huge, gitignored) field data.
Handles restarts by concatenating every time directory and de-duplicating
overlapping samples.

    python3 tools/analyze.py [--case DIR] [--tmin T] [--no-plots]
"""
import argparse, glob, os, re, sys
import numpy as np

# ---- cell / bath constants (must match the case setup) --------------------
Q_GEN      = 18.2           # W, ohmic + entropic at 35 A (baseline input); main() replaces it by the case's own value
D_CELL     = 0.018
H_CELL     = 0.065
V_CELL     = np.pi * (D_CELL / 2) ** 2 * H_CELL
A_CELL     = np.pi * D_CELL * H_CELL + 2 * np.pi * (D_CELL / 2) ** 2   # can + both ends
RHO_CELL, CP_CELL = 2700.0, 1050.0
C_CELL     = RHO_CELL * V_CELL * CP_CELL            # J/K, whole cell
V_BATH     = 0.060 * 0.060 * 0.130 - V_CELL         # m3, whole bath
T_AMB      = 308.15
QUARTER    = 4.0            # the mesh is one quadrant

def _read(path, ncols=None):
    """Read an OpenFOAM postProcessing .dat, skipping '#' headers."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.split())
    if not rows:
        return None
    n = ncols or min(len(r) for r in rows)
    return np.array([[float(v) for v in r[:n]] for r in rows])

def series(case, region, name, fname, col=1):
    """Concatenate one monitor across all restart directories.

    Note the two different layouts OpenFOAM uses in a multi-region run:
    volFieldValue/surfaceFieldValue write to postProcessing/<region>/<name>/,
    but probes write to postProcessing/<name>/<region>/.
    """
    pat = os.path.join(case, "postProcessing", region, name, "*", fname)
    if not glob.glob(pat):
        pat = os.path.join(case, "postProcessing", name, region, "*", fname)
    parts = []
    for p in sorted(glob.glob(pat), key=lambda s: float(os.path.basename(os.path.dirname(s)))):
        a = _read(p)
        if a is not None and a.shape[1] > col:
            parts.append(a[:, [0, col]])
    if not parts:
        return None
    a = np.vstack(parts)
    a = a[np.argsort(a[:, 0])]
    _, keep = np.unique(np.round(a[:, 0], 9), return_index=True)   # drop restart overlap
    return a[keep]

def heat_rates(case):
    """wallHeatFlux.dat -> {patch: (t, W)} for the quarter domain, scaled to full."""
    out = {}
    for p in sorted(glob.glob(os.path.join(case, "postProcessing", "fluid",
                                           "wallHeatFlux", "*", "wallHeatFlux.dat"))):
        with open(p) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                c = line.split()
                # "<time> <patch> <min> <max> <integral>"
                try:
                    t = float(c[0]); patch = c[1]; integral = float(c[-1])
                except (ValueError, IndexError):
                    continue
                out.setdefault(patch, []).append((t, integral * QUARTER))
    return {k: np.array(sorted(set(v))) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cell18650_immersion")
    ap.add_argument("--tmin", type=float, default=0.0, help="ignore data before this time")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--outdir", default="plots")
    a = ap.parse_args()
    # heat rate actually applied in this case (system/battery/fvOptions), if the case carries one
    global Q_GEN
    fo = os.path.join(a.case, "system", "battery", "fvOptions")
    if os.path.isfile(fo):
        m = re.search(r"^\s*h\s+\(\s*([0-9.eE+-]+)\s+[0-9.eE+-]+\s*\)\s*;", open(fo).read(), re.M)
        if m:
            Q_GEN = float(m.group(1)) * 1.65407e-5      # W/m3 x nominal cell volume

    M = {
        "cell_max":  series(a.case, "battery", "cellTemperatureMax",     "volFieldValue.dat"),
        "cell_mean": series(a.case, "battery", "cellTemperatureMean",    "volFieldValue.dat"),
        "bath_mean": series(a.case, "fluid",   "bathTemperatureMean",    "volFieldValue.dat"),
        "bath_max":  series(a.case, "fluid",   "bathTemperatureMax",     "volFieldValue.dat"),
        "u_max":     series(a.case, "fluid",   "plumeSpeed",             "volFieldValue.dat"),
        "can_surf":  series(a.case, "fluid",   "canSurfaceTemperature",  "surfaceFieldValue.dat"),
    }
    missing = [k for k, v in M.items() if v is None]
    if missing:
        print("WARNING: no data for {}".format(", ".join(missing)), file=sys.stderr)
    if M["cell_max"] is None:
        sys.exit("no monitor data found under {}/postProcessing".format(a.case))

    t = M["cell_max"][:, 0]
    sel = t >= a.tmin
    C = lambda k: (M[k][:, 1] - 273.15) if M[k] is not None else None

    def interp(k):
        if M[k] is None:
            return None
        return np.interp(t, M[k][:, 0], M[k][:, 1])

    Tmax, Tmean = interp("cell_max"), interp("cell_mean")
    Tbath, Tbmax = interp("bath_mean"), interp("bath_max")
    Tcan = interp("can_surf")
    umax = interp("u_max")

    # Heat actually leaving the cell = generated minus what the cell stores.
    # Differentiating the mean cell temperature is noisy, so smooth it lightly.
    dT = np.gradient(Tmean, t) if len(t) > 3 else np.zeros_like(t)
    if len(t) > 21:
        w = np.ones(11) / 11.0
        dT = np.convolve(dT, w, mode="same")
    Q_out = Q_GEN - C_CELL * dT

    # Immersion heat transfer coefficient on the can, referenced to bulk bath.
    dTfilm = (Tcan - Tbath) if Tcan is not None else None
    # h is meaningless while the film dT is within noise of zero (start-up) and
    # at the last few samples (edge of the smoothed dT/dt) -- mask both.
    h = None
    if dTfilm is not None:
        h = np.where(dTfilm > 0.5, Q_out / (A_CELL * np.maximum(dTfilm, 1e-6)), np.nan)
        if len(h) > 6:
            h[-3:] = np.nan

    HR = heat_rates(a.case)

    print("=" * 78)
    print("18650 at 35 A ({:.1f} W) in a {:.0f} mL ester bath, {:.0f} degC room"
          .format(Q_GEN, V_BATH * 1e6, T_AMB - 273.15))
    print("=" * 78)
    print("simulated {:.1f} s of a 267.4 s discharge ({:.0f} %), {:d} samples"
          .format(t[-1], 100 * t[-1] / 267.4, len(t)))
    print()
    hdr = "  {:>7s} {:>9s} {:>9s} {:>9s} {:>9s} {:>8s} {:>8s} {:>8s}"
    print(hdr.format("t [s]", "T_core", "T_cell", "T_can", "T_bath", "dT_film", "h", "u_max"))
    print(hdr.format("", "[degC]", "[degC]", "[degC]", "[degC]", "[K]", "[W/m2K]", "[m/s]"))
    idx = [int(x) for x in np.linspace(0, len(t) - 1, min(len(t), 14))]
    for i in idx:
        print("  {:7.1f} {:9.2f} {:9.2f} {:9.2f} {:9.2f} {:8.2f} {:8.1f} {:8.4f}".format(
            t[i], Tmax[i] - 273.15, Tmean[i] - 273.15,
            (Tcan[i] - 273.15) if Tcan is not None else np.nan,
            Tbath[i] - 273.15,
            dTfilm[i] if dTfilm is not None else np.nan,
            h[i] if h is not None else np.nan,
            umax[i] if umax is not None else np.nan))

    print("\n  end of run:")
    print("    cell hot spot          {:7.2f} degC   (rise {:+.2f} K)".format(Tmax[-1]-273.15, Tmax[-1]-T_AMB))
    print("    cell volume mean       {:7.2f} degC".format(Tmean[-1]-273.15))
    if Tcan is not None:
        print("    can surface (mean)     {:7.2f} degC".format(Tcan[-1]-273.15))
        print("    core-to-surface drop   {:7.2f} K      (radial conduction in the jellyroll)"
              .format(Tmax[-1]-Tcan[-1]))
    print("    bath bulk              {:7.2f} degC   (rise {:+.2f} K)".format(Tbath[-1]-273.15, Tbath[-1]-T_AMB))
    print("    bath hot spot          {:7.2f} degC".format(Tbmax[-1]-273.15))
    if h is not None:
        print("    immersion h            {:7.1f} W/m2K".format(h[-1]))
    if umax is not None:
        print("    peak plume speed       {:7.4f} m/s".format(umax[-1]))

    # ---- energy balance: generated = cell stored + bath stored + wall loss ----
    print("\n  energy balance over 0 -> {:.1f} s:".format(t[-1]))
    E_gen = Q_GEN * (t[-1] - t[0])
    E_cell = C_CELL * (Tmean[-1] - Tmean[0])
    rho_b, cp_b = 950.0, 2000.0                    # bath mean values over the range
    E_bath = V_BATH * rho_b * cp_b * (Tbath[-1] - Tbath[0])
    print("    generated              {:9.1f} J".format(E_gen))
    print("    stored in cell         {:9.1f} J  ({:5.1f} %)".format(E_cell, 100*E_cell/E_gen))
    print("    stored in bath         {:9.1f} J  ({:5.1f} %)".format(E_bath, 100*E_bath/E_gen))
    resid = E_gen - E_cell - E_bath
    print("    residual (wall loss +")
    print("      integration error)   {:9.1f} J  ({:5.2f} %)".format(resid, 100*resid/E_gen))

    if HR:
        print("\n  boundary heat rates, last sample (scaled to the full cell):")
        for p, arr in sorted(HR.items()):
            print("    {:<20s} {:8.2f} W".format(p, arr[-1, 1]))

    if a.no_plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.outdir, exist_ok=True)

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax[0].plot(t, Tmax-273.15,  label="cell hot spot")
    ax[0].plot(t, Tmean-273.15, label="cell mean")
    if Tcan is not None:
        ax[0].plot(t, Tcan-273.15, label="can surface")
    ax[0].plot(t, Tbmax-273.15, ls="--", label="bath hot spot")
    ax[0].plot(t, Tbath-273.15, ls="--", label="bath bulk")
    ax[0].axhline(60, color="r", lw=0.8, ls=":", label="60 degC design limit")
    ax[0].set_ylabel("temperature [degC]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("18650 at 35 A (18.2 W), immersed in {:.0f} mL dielectric ester, 35 degC room"
                    .format(V_BATH*1e6), fontsize=10)

    if h is not None:
        ax[1].plot(t, h, color="C2")
        ax[1].set_ylabel("immersion h [W/m2K]"); ax[1].grid(alpha=.3)
        ax[1].set_ylim(bottom=0)
    if umax is not None:
        a2 = ax[1].twinx(); a2.plot(t, umax, color="C3", lw=.9, label="peak speed")
        a2.set_ylabel("peak fluid speed [m/s]", color="C3")

    ax[2].plot(t, Q_out, label="out of the cell")
    ax[2].axhline(Q_GEN, color="k", lw=.8, ls=":", label="generated (18.2 W)")
    for p, arr in sorted(HR.items()):
        if "battery" not in p:
            ax[2].plot(arr[:, 0], -arr[:, 1], lw=.9, ls="--", label=p)
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("heat rate [W]")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.tight_layout()
    out = os.path.join(a.outdir, "thermal_history.png")
    fig.savefig(out, dpi=140)
    print("\n  wrote {}".format(out))

if __name__ == "__main__":
    main()
