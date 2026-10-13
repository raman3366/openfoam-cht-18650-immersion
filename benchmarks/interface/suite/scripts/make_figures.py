#!/usr/bin/env python3
"""
make_figures.py -- figures and the acceptance-test table from the case results.

    make_figures.py [<casesDir> [<figDir>]]

Output (PDF, 3.4 inch wide, serif, no titles; fonts embedded as TrueType):
    fig1_error_vs_angle_ratio.pdf   patch test: largest temperature error against angle and anisotropy ratio
    fig1b_error_map.pdf             the same error over the whole ratio x angle grid, one panel per coupling
    fig1c_sheared_closed_form.pdf   sheared strip: wall-temperature error of the stock directional option against
                                    its closed form
    fig2_flux_ratio_mesh.pdf        heat received over heat released against mesh size
    fig3_order_annulus.pdf          error against mesh size: aligned annulus and ring with oblique principal axes
    fig4_staggering_dt.pdf          staggering error and energy drift against time step, 1 to 3 coupling passes
    table_acceptance.tex / .csv     acceptance-test measures, conserving and non-conserving configurations
Every plotted number is read from <case>/result.json; figdata_*.csv holds the plotted values.
Colours: first three slots of a palette checked for colour-vision deficiency (all pairs);
series are also distinguished by marker and line style, so the figures read in grey scale.
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact as ex                       # noqa: E402

BENCH = os.path.dirname(HERE)
W = 3.4
INK, INK2, GRID = '#0b0b0b', '#52514e', '#dddcd8'
COL = dict(scalar='#2a78d6', directional='#eb6834', corrected='#1baf7a')
MRK = dict(scalar='o', directional='s', corrected='^')
LAB = dict(scalar='scalar option', directional='directional option', corrected='co-normal correction')
RATIOS = (1, 3, 10, 35, 100, 200, 1000)
ANGLES = (0, 15, 30, 45, 60, 75, 90)
FLOOR = 1e-13

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'lines.linewidth': 1.1, 'lines.markersize': 4.2, 'grid.color': GRID, 'grid.linewidth': 0.4,
    'legend.frameon': False, 'legend.handlelength': 2.2, 'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600, 'figure.dpi': 150,
})


class Data:
    def __init__(self, root):
        self.root, self.cache = root, {}

    def __call__(self, cid):
        if cid not in self.cache:
            f = os.path.join(self.root, cid, 'result.json')
            self.cache[cid] = json.load(open(f)) if os.path.exists(f) else None
        return self.cache[cid]

    def diverged(self, cid):
        return os.path.exists(os.path.join(self.root, cid, 'aborted.json'))


def panel(ax, tag):
    ax.text(0.03, 0.95, tag, transform=ax.transAxes, va='top', ha='left', fontsize=8, color=INK)


def style(ax, logx=False, logy=False):
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.grid(True, which='major')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def series(ax, x, y, key, label=None, ls='-', filled=True, **kw):
    y = np.maximum(np.asarray(y, float), FLOOR) if kw.pop('clip', True) else np.asarray(y, float)
    return ax.plot(x, y, ls, color=COL[key], marker=MRK[key], mfc=(COL[key] if filled else 'white'), mec=COL[key],
                   mew=0.9, label=label if label is not None else LAB[key], **kw)[0]


def save(fig, figdir, name, rows=None, header=None):
    path = os.path.join(figdir, name + '.pdf')
    fig.savefig(path)
    plt.close(fig)
    if rows:
        with open(os.path.join(figdir, 'figdata_' + name + '.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    print('wrote', path)


def cpl_cases(kind, r, th, n=16):
    """Case ids of the three couplings for one P1 point (corrected = derived BC of this job)."""
    if kind == 'patch':
        return {'scalar': f'p1_r{r}_t{th}_scalar_n{n}', 'directional': f'p1_r{r}_t{th}_directional_n{n}',
                'corrected': f'p1_r{r}_t{th}_conormalbc_n{n}'}
    return {'scalar': f'p1s_r{r}_t{th}_scalar', 'directional': f'p1s_r{r}_t{th}_directional',
            'corrected': f'p1s_r{r}_t{th}_conormalbc'}


# ------------------------------------------------------------------ figure 1


def fig1(D, figdir):
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.1), constrained_layout=True)
    rows = []
    for ax, xs, fixed, xl, tag in ((axs[0], ANGLES, 35, r'angle $\theta$ between principal axis 1 and the normal (deg)',
                                    r'(a) $k_1/k_2=35$'),
                                   (axs[1], RATIOS, 45, r'anisotropy ratio $k_1/k_2$', r'(b) $\theta=45^\circ$')):
        for key in ('scalar', 'directional', 'corrected'):
            X, Y, Xd = [], [], []
            for x in xs:
                r, th = (fixed, x) if ax is axs[0] else (x, fixed)
                cid = cpl_cases('patch', r, th)[key]
                d = D(cid)
                if d:
                    X.append(x); Y.append(d['error']['exact']['all']['Linf'])
                    rows.append((tag[:3], key, r, th, Y[-1]))
                elif D.diverged(cid):
                    Xd.append(x)
                    rows.append((tag[:3], key, r, th, 'diverged'))
            if X:
                series(ax, X, Y, key)
            if Xd:              # pseudo-time march diverged: no steady state to plot
                ax.plot(Xd, [3e-11] * len(Xd), ls='', marker='x', color=COL[key], ms=5, mew=1.2)
                ax.annotate('march diverged', (Xd[0], 3e-11), xytext=(0, 7), textcoords='offset points', ha='center',
                            fontsize=6.5, color=INK2)
        style(ax, logx=(ax is axs[1]), logy=True)
        ax.set_xlabel(xl)
        ax.set_ylabel('largest temperature error (K)')
        ax.set_ylim(1e-13, 3e2)
        ax.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1e0])
        panel(ax, tag)
        if ax is axs[0]:
            ax.set_xticks(ANGLES)
    axs[0].legend(loc='center right', bbox_to_anchor=(1.0, 0.42))
    save(fig, figdir, 'fig1_error_vs_angle_ratio', rows, ('panel', 'coupling', 'ratio', 'theta_deg', 'Linf_error_K'))


def fig1b(D, figdir):
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.75), constrained_layout=True, sharey=True)
    rows, im = [], None
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#2a78d6', '#0d2f5c'])
    for ax, key in zip(axs, ('scalar', 'directional', 'corrected')):
        Z = np.full((len(RATIOS), len(ANGLES)), np.nan)
        for i, r in enumerate(RATIOS):
            for j, th in enumerate(ANGLES):
                cid = cpl_cases('patch', r, th)[key]
                d = D(cid)
                if d:
                    Z[i, j] = max(d['error']['exact']['all']['Linf'], FLOOR)
                    rows.append((key, r, th, d['error']['exact']['all']['Linf']))
                elif D.diverged(cid):
                    ax.plot([j], [i], marker='x', color=INK2, ms=4, mew=0.9)
                    rows.append((key, r, th, 'diverged'))
        im = ax.imshow(np.log10(Z), origin='lower', cmap=cmap, vmin=-12, vmax=1.5, aspect='auto')
        ax.set_xticks(range(0, len(ANGLES), 2)); ax.set_xticklabels([ANGLES[j] for j in range(0, len(ANGLES), 2)])
        ax.set_yticks(range(len(RATIOS))); ax.set_yticklabels(RATIOS)
        ax.set_xlabel(r'$\theta$ (deg)')
        ax.tick_params(length=0)
        ax.text(0.5, 1.03, LAB[key].replace(' option', '').replace('co-normal correction', 'corrected'),
                transform=ax.transAxes, ha='center', va='bottom', fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel(r'$k_1/k_2$')
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02, aspect=18)
    cb.set_label(r'$\log_{10}$ error (K)', fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=6.5)
    save(fig, figdir, 'fig1b_error_map', rows, ('coupling', 'ratio', 'theta_deg', 'Linf_error_K'))


def fig1c(D, figdir):
    fig, ax = plt.subplots(figsize=(W, 2.3), constrained_layout=True)
    rows = []
    th_f = np.linspace(0, 90, 181)
    shades = {3: '#f4b8a0', 35: '#eb6834', 1000: '#8f3312'}
    for r in (3, 35, 1000):
        k = (0.8 * r, 0.8, 0.8)
        cf = [ex.StripSteady.from_gradient(k, t, 1.0, 0.01, 0.01, 320.0, 1000.0, 500.0) for t in th_f]
        ax.plot(th_f, [s.scheme(s.Knn)['Tw'] - s.exact()['Tw'] for s in cf], '-', color=shades[r], lw=0.9)
        X, Y = [], []
        for th in ANGLES:
            d = D(f'p1s_r{r}_t{th}_directional')
            if d:
                P = d['params']
                X.append(th); Y.append(d['interface']['Tw_aniso'] - P['B'] * P['H'] / 2 - d['exact']['Tw'])
                rows.append((r, th, Y[-1], d['exact']['scheme_wall_error_sheared']))
        ax.plot(X, Y, 's', color=shades[r], mfc=shades[r], mec=shades[r], ms=4)
        if len(X) == len(ANGLES):          # direct labels placed clear of the other curves
            j, off, ha = {3: (5, (0, 5), 'center'), 35: (4, (2, -11), 'center'), 1000: (5, (-5, 5), 'right')}[r]
            ax.annotate(rf'$k_1/k_2={r}$', (X[j], Y[j]), xytext=off, textcoords='offset points', ha=ha,
                        fontsize=7, color=INK)
    style(ax)
    ax.set_xticks(ANGLES)
    ax.set_xlabel(r'angle $\theta$ (deg)')
    ax.set_ylabel('wall temperature error (K)')
    ax.legend(handles=[Line2D([], [], color=INK2, lw=0.9, label='closed form'),
                       Line2D([], [], color=INK2, marker='s', ls='', ms=4, label='solver')], loc='upper left')
    save(fig, figdir, 'fig1c_sheared_closed_form', rows, ('ratio', 'theta_deg', 'wall_error_measured_K', 'wall_error_closed_form_K'))


# ------------------------------------------------------------------ figure 2


def fig2(D, figdir):
    pred = ex.frob((28.0, 0.8, 0.8)) / 0.8
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.0), constrained_layout=True, sharex=True)
    rows = []
    ns = (5, 10, 20, 40, 80, 160)
    raw, des, four = [], [], []
    for n in ns:
        d1, d4 = D(f'xm_slab_scalar_t90_n{n}_o1'), D(f'xm_slab_scalar_t90_n{n}_o4')
        raw.append(d1['audit']['ratio_received_over_released'] if d1 else np.nan)
        des.append(d1['audit']['destaggered_ratio'] if d1 else np.nan)
        four.append(d4['audit']['ratio_received_over_released'] if d4 else np.nan)
        rows.append(('slab', n, raw[-1], des[-1], four[-1], pred))
    an_n, an = [], []
    for lev in (1, 2, 4, 8):
        d = D(f'p2A_r35_scalar_L{lev}')
        if d:
            m = d['measured_Qp']
            an_n.append(10 * lev); an.append(m['interface_into_iso'] / m['interface_out_of_aniso'])
            rows.append(('annulus', 10 * lev, '', '', an[-1], pred))
    ax = axs[0]
    ax.axhline(pred, color=INK2, lw=0.7, ls=(0, (4, 2)))
    ax.plot(ns, raw, '-', color=COL['directional'], marker='s', mfc='white', mec=COL['directional'], mew=0.9,
            label='transient slab, one pass per step')
    ax.plot(ns, four, '-', color=COL['scalar'], marker='o', label='transient slab, four passes')
    ax.plot(an_n, an, color=COL['corrected'], marker='^', ls='', label='steady annulus')
    style(ax, logx=True)
    ax.set_ylabel('heat received / heat released')
    ax.set_ylim(pred - 0.05, pred + 0.2)
    ax.annotate(r'$\Vert K\Vert_F/(n\!\cdot\!K\!\cdot\!n)=35.0286$', (5.2, pred), xytext=(0, -9), textcoords='offset points',
                fontsize=7, color=INK2)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88))
    panel(ax, '(a)')
    ax = axs[1]
    ax.plot(ns, np.abs(np.array(des) / pred - 1), '-', color=COL['directional'], marker='s', mfc='white',
            mec=COL['directional'], mew=0.9, label='one pass, staggering term removed')
    ax.plot(ns, np.abs(np.array(four) / pred - 1), '-', color=COL['scalar'], marker='o', label='four passes')
    ax.plot(an_n, np.abs(np.array(an) / pred - 1), color=COL['corrected'], marker='^', ls='', label='steady annulus')
    style(ax, logx=True, logy=True)
    ax.set_ylim(1e-14, 1e-3)
    ax.set_xticks(ns); ax.set_xticklabels(ns); ax.minorticks_off()
    ax.set_xlabel('cells across the anisotropic region')
    ax.set_ylabel('|measured / predicted $-$ 1|')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.88))
    panel(ax, '(b)')
    save(fig, figdir, 'fig2_flux_ratio_mesh', rows, ('problem', 'cells_across_solid', 'ratio_one_pass_raw',
                                                     'ratio_one_pass_destaggered', 'ratio_four_passes_or_steady', 'predicted'))


# ------------------------------------------------------------------ figure 3


def _order(h, e):
    return [math.log(e[i] / e[i + 1]) / math.log(h[i] / h[i + 1]) for i in range(len(e) - 1)]


def fig3(D, figdir):
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.6), constrained_layout=True, sharex=True)
    rows, orders = [], {}

    def curve(ax, ids, key, which='exact', norm='L2', **kw):
        H, E = [], []
        for lev, cid in ids:
            d = D(cid)
            if d and which in d['error']:
                H.append(5.0 / (10 * lev)); E.append(d['error'][which]['all'][norm])
                rows.append((cid, key, which, norm, H[-1], E[-1], d['error'][which]['all']['Linf']))
        if H:
            series(ax, H, E, key, **kw)
            if len(H) > 1:
                orders[(ids[0][1].rsplit('_L', 1)[0], which)] = _order(H, E)
        return H, E

    ax = axs[0]
    lv = (1, 2, 4, 8)
    curve(ax, [(l, f'p2A_r35_scalar_L{l}') for l in lv], 'scalar', label='scalar, against exact')
    curve(ax, [(l, f'p2A_r35_scalar_L{l}') for l in lv], 'scalar', which='scheme', ls='--', filled=False,
          label='scalar, against its own problem')
    H, E = curve(ax, [(l, f'p2A_r35_directional_L{l}') for l in lv], 'directional', label='directional')
    curve(ax, [(l, f'p2A_r35_conormalbc_L{l}') for l in lv], 'corrected', ls=':', filled=False, label='co-normal correction')
    if H:
        hh = np.array([H[0], H[-1]])
        ax.plot(hh, E[0] * 0.45 * (hh / H[0]) ** 2, '-', color=INK2, lw=0.6)
        ax.annotate('slope 2', (hh[-1], E[0] * 0.45 * (hh[-1] / H[0]) ** 2), xytext=(-2, -9), textcoords='offset points',
                    ha='right', fontsize=7, color=INK2)
    style(ax, logx=True, logy=True)
    ax.set_ylabel(r'$L_2$ temperature error (K)')
    ax.set_ylim(1e-4, 1e3)
    ax.legend(loc='lower center', bbox_to_anchor=(0.46, 1.02), ncol=2, columnspacing=1.0, handletextpad=0.5)
    panel(ax, '(a) normal along a principal axis')

    ax = axs[1]
    lv = (1, 2, 3, 4, 6)
    curve(ax, [(l, f'ring_r35_q_scalar_L{l}') for l in (1, 2)], 'scalar')
    curve(ax, [(l, f'ring_r35_q_directional_L{l}') for l in lv], 'directional')
    H, E = curve(ax, [(l, f'ring_r35_q_conormalbc_L{l}') for l in lv], 'corrected')
    if H:
        hh = np.array([H[0], H[-1]])
        ax.plot(hh, E[0] * 0.45 * (hh / H[0]) ** 2, '-', color=INK2, lw=0.6)
        ax.annotate('slope 2', (hh[-1], E[0] * 0.45 * (hh[-1] / H[0]) ** 2), xytext=(-2, -9), textcoords='offset points',
                    ha='right', fontsize=7, color=INK2)
    style(ax, logx=True, logy=True)
    ax.set_ylim(1e-4, 1e3)
    ax.set_xlabel('radial cell size in the anisotropic region (mm)')
    ax.set_ylabel(r'$L_2$ temperature error (K)')
    ax.set_xticks([0.5, 0.25, 0.125, 0.0625]); ax.set_xticklabels(['0.5', '0.25', '0.125', '0.0625']); ax.minorticks_off()
    ax.invert_xaxis()
    panel(ax, '(b) normal oblique to the principal axes')
    save(fig, figdir, 'fig3_order_annulus', rows, ('case', 'coupling', 'against', 'plotted_norm', 'h_mm', 'error_K', 'Linf_K'))
    json.dump({f'{k[0]}|{k[1]}': v for k, v in orders.items()}, open(os.path.join(figdir, 'figdata_fig3_orders.json'), 'w'),
              indent=1)
    return orders


# ------------------------------------------------------------------ figure 4


def fig4(D, figdir, lev=2):
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.0), constrained_layout=True, sharex=True)
    rows = []
    keys = {1: 'directional', 2: 'scalar', 3: 'corrected'}       # colour slots only; labels say passes
    lab = {1: 'one pass per step', 2: 'two passes', 3: 'three passes'}
    dts = [0.4 / m for m in (1, 2, 4, 8)]
    first = {}
    for no in (1, 2, 3):
        S, Dr = [], []
        for m in (1, 2, 4, 8):
            d = D(f'p3_directional_o{no}_L{lev}_dt{m}')
            if d:
                a = d['audit']
                S.append(a['split_Linf_K']['staggering__emulator_minus_monolithic'])
                Dr.append(abs(a['drift']) / abs(a['dE_aniso']))
                rows.append((no, 0.4 / m, S[-1], Dr[-1], a['split_Linf_K']['euler__monolithic_minus_semidiscrete'],
                             a['split_Linf_K']['solver_minus_emulator']))
            else:
                S.append(np.nan); Dr.append(np.nan)
        k = keys[no]
        axs[0].plot(dts, S, '-', color=COL[k], marker=MRK[k], mew=0.9, label=lab[no])
        axs[1].plot(dts, Dr, '-', color=COL[k], marker=MRK[k], mew=0.9, label=lab[no])
        first[no] = (S, Dr)
    eul = []
    for m in (1, 2, 4, 8):
        d = D(f'p3_directional_o1_L{lev}_dt{m}')
        eul.append(d['audit']['split_Linf_K']['euler__monolithic_minus_semidiscrete'] if d else np.nan)
    axs[0].plot(dts, eul, color=INK2, lw=0.8, ls=(0, (4, 2)), label='implicit-Euler error')
    for ax, (S, _), j in ((axs[0], first[1], 0), (axs[1], first[1], 1)):
        y0 = first[1][j][0]
        if y0 == y0:
            hh = np.array([dts[0], dts[-1]])
            ax.plot(hh, 2.2 * y0 * hh / dts[0], '-', color=INK2, lw=0.6)
            ax.annotate('slope 1', (hh[-1], 2.2 * y0 * hh[-1] / dts[0]), xytext=(2, 4), textcoords='offset points', fontsize=7,
                        color=INK2)
    for ax in axs:
        style(ax, logx=True, logy=True)
    axs[0].set_ylabel('staggering error (K)')
    axs[1].set_ylabel('energy drift / energy exchanged')
    axs[1].set_xlabel('time step (s)')
    axs[1].set_xticks(dts); axs[1].set_xticklabels(['0.4', '0.2', '0.1', '0.05']); axs[1].minorticks_off()
    axs[0].legend(loc='lower right', ncol=1)
    panel(axs[0], '(a)'); panel(axs[1], '(b)')
    save(fig, figdir, 'fig4_staggering_dt', rows, ('passes', 'dt_s', 'staggering_Linf_K', 'relative_energy_drift',
                                                   'implicit_Euler_Linf_K', 'solver_minus_emulator_K'))


# ------------------------------------------------------------------ acceptance table


def acceptance(D, figdir):
    """Four configurations, each on the same two problems: closed slab (storage audit) and
    oblique patch test (two-sided steady balance)."""
    cfg = [('scalar option', 'not conserving', 'xd_aud_scalar_t90_o4', 'xaud_p1_t30_scalar'),
           ('directional, aligned', 'conserving', 'xd_aud_directional_t90_o4', 'xaud_p1_t90_directional'),
           ('directional, oblique', 'not conserving', None, 'xaud_p1_t30_directional'),
           ('co-normal, oblique', 'conserving', None, 'xaud_p1_t30_conormalbc')]
    rows = []
    for name, verdict, slab, patch in cfg:
        row = dict(configuration=name, verdict=verdict)
        d = D(slab) if slab else None
        if d:
            a = d['audit']
            fo = a.get('function_objects', {})
            row['storage_ratio'] = a['ratio_received_over_released']
            row['stock_books_ratio'] = -fo['stock_iso']['integral'] / fo['stock_aniso']['integral'] if fo.get('stock_aniso') else ''
            row['corrected_books_ratio'] = (-fo['stock_iso']['integral'] / fo['conormal_aniso_total']['integral']
                                            if fo.get('conormal_aniso_total') else '')
        p = D(patch)
        if p:
            i = p['interface']
            ex_rate = abs(p['exact']['rate_plus_xi'])
            row['patch_error_K'] = p['error']['exact']['all']['Linf']
            row['created_over_exact_rate'] = i['imbalance'] / ex_rate
            if 'stock_wallHeatFlux_aniso' in i:
                row['stock_two_sided_residual'] = (i['stock_wallHeatFlux_aniso'] + i['stock_wallHeatFlux_iso']) / ex_rate
            if 'coNormalWallHeatFlux_aniso' in i:
                row['corrected_two_sided_residual'] = (i['coNormalWallHeatFlux_aniso'] + i['stock_wallHeatFlux_iso']) / ex_rate
        rows.append(row)
    cols = ['configuration', 'verdict', 'storage_ratio', 'stock_books_ratio', 'corrected_books_ratio', 'patch_error_K',
            'created_over_exact_rate', 'stock_two_sided_residual', 'corrected_two_sided_residual']
    with open(os.path.join(figdir, 'table_acceptance.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def f(v, fmt):
        if v == '' or v is None:
            return '--'
        if fmt == 'e':
            if abs(v) < 5e-9:
                return r'$<10^{-8}$'
            return f'{v:.3f}' if abs(v) >= 1e-3 else f'${v:.1e}'.replace('e-0', r'\times10^{-').replace('e-', r'\times10^{-') + '}$'
        return f'{v:{fmt}}'
    lines = [r'% generated by scripts/make_figures.py from the case results; do not edit by hand',
             r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             r' & scalar & directional, & directional, & co-normal, \\',
             r'Measure & option & aligned & oblique & oblique \\', r'\midrule']
    meas = [('Closed pair, storage audit: heat received / released', 'storage_ratio', '.6f'),
            (r'Same ratio from the stock \texttt{wallHeatFlux} objects', 'stock_books_ratio', '.6f'),
            (r'Same ratio with the tensor flux on the solid side', 'corrected_books_ratio', '.6f'),
            (r'Patch test, largest temperature error (K)', 'patch_error_K', 'e'),
            (r'Energy created at the interface / exact rate', 'created_over_exact_rate', 'e'),
            (r'Two-sided residual, stock objects / exact rate', 'stock_two_sided_residual', 'e'),
            (r'Two-sided residual, tensor flux / exact rate', 'corrected_two_sided_residual', 'e')]
    for label, key, fmt in meas:
        lines.append(label + ' & ' + ' & '.join(f(r.get(key, ''), fmt) for r in rows) + r' \\')
    lines += [r'\midrule', 'Conserving & ' + ' & '.join('yes' if r['verdict'] == 'conserving' else 'no' for r in rows) + r' \\',
              r'\bottomrule', r'\end{tabular}']
    open(os.path.join(figdir, 'table_acceptance.tex'), 'w').write('\n'.join(lines) + '\n')
    print('wrote table_acceptance.tex/.csv')
    return rows


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BENCH, 'cases')
    figdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BENCH, 'figures')
    os.makedirs(figdir, exist_ok=True)
    D = Data(root)
    todo = sys.argv[3:] or ['fig1', 'fig1b', 'fig1c', 'fig2', 'fig3', 'fig4', 'acceptance']
    for name in todo:
        try:
            globals()[name](D, figdir)
        except Exception as e:           # a figure whose runs are not finished yet
            import traceback
            print(f'{name}: FAILED {e!r}')
            traceback.print_exc()
