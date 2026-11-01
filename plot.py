"""Figures and summaries for the H^2 illustration (Section 6).
Usage: python3 plot.py OUTPREFIX results1.npz [results2.npz ...] [--fitmin N] [--cmap NAME LIGHT DARK]
Writes OUTPREFIX.pdf/.png (L^1 and L^alpha risks), OUTPREFIX_mc.pdf/.png (Monte Carlo diagnostic
for the L^alpha risk at the largest n), OUTPREFIX_summary.csv (fitted slopes), OUTPREFIX_risks.csv."""
import json, sys, csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

args = sys.argv[1:]
if '--cmap' in args:
    i = args.index('--cmap'); del args[i:i + 4]
fitmin = 64
if '--fitmin' in args:
    i = args.index('--fitmin'); fitmin = int(args[i + 1]); del args[i:i + 2]
out, files = args[0], args[1:]
nu = 1.5
res, info = {}, {}
for f in files:
    z = np.load(f)
    inf = json.loads(str(z['info']))
    for k in z.files:
        if k.startswith('err_'):
            _, a, n = k.split('_'); key = (float(a), int(n))
            res[key] = z[k].astype(float)
            info[key] = inf.get(f'{a}_{n}', dict(max_iter=0, not_converged=0))
alphas = sorted({a for a, _ in res}); ns = sorted({n for _, n in res}); nn = np.array(ns, float)

# ordinal color scale in alpha: a perceptually uniform, colorblind-safe colormap, light -> dark with increasing
# alpha; the lightest end is cut off so that all lines remain visible on white
CMAP, LIGHT, DARK = 'plasma', 0.80, 0.0
for i, x in enumerate(sys.argv):
    if x == '--cmap': CMAP, LIGHT, DARK = sys.argv[i + 1], float(sys.argv[i + 2]), float(sys.argv[i + 3])
from matplotlib.colors import to_hex
COL = {a: to_hex(plt.get_cmap(CMAP)(x)) for a, x in zip(alphas, np.linspace(LIGHT, DARK, len(alphas)))}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8.5, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6})

def pred(a): return -0.5 if 2 * a - 2 < nu else -(1 - (a - 1) / nu)
sel = nn >= fitmin
def slope(y): return np.polyfit(np.log(nn[sel]), np.log(y[sel]), 1)[0]

tab, rows = {}, []
for a in alphas:
    L1, La, med = [], [], []
    for n in ns:
        e = res[(a, n)]; R = len(e)
        L1.append(e.mean()); La.append(np.mean(e ** a) ** (1 / a)); med.append(np.median(e))
        rows.append((a, n, R, e.mean(), e.std() / np.sqrt(R), np.mean(e ** a) ** (1 / a), np.median(e),
                     np.quantile(e, .9), e.max(), info[(a, n)]['max_iter'], info[(a, n)]['not_converged'],
                     info[(a, n)].get('recomputed_high_precision', 0)))
    tab[a] = dict(L1=np.array(L1), La=np.array(La), med=np.array(med))

def style_ax(ax, title, ylab):
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_title(title, fontsize=9, color=INK, loc='left'); ax.set_ylabel(ylab)
    ax.grid(True, which='major', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(['1', '4', '16', '64', '256', '1024'])
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
small = [a for a in alphas if a < nu]
for a in alphas:
    axs[0].plot(nn, tab[a]['L1'], '-', marker='o', ms=2.6, lw=1.3, color=COL[a])
    if a < nu: axs[1].plot(nn, tab[a]['La'], '-', marker='o', ms=2.6, lw=1.3, color=COL[a])
for ax, key, al in [(axs[0], 'L1', alphas), (axs[1], 'La', small)]:
    yend = min(tab[a][key][-1] for a in al)
    ref = yend * 0.55 * (nn / nn[-1]) ** -0.5
    ax.plot(nn, ref, '-', color=MUTED, lw=0.7)
    ax.annotate(r'slope $-1/2$', xy=(nn[-4], ref[-4]), xytext=(-4, -4), textcoords='offset points',
                fontsize=7.5, color=MUTED, ha='right', va='top')
    ax.set_xlabel('sample size $n$')
style_ax(axs[0], r'(a) $L^1$-risk', r'$\mathbf{E}[\,\overline{m m_n}\,]$')
style_ax(axs[1], r'(b) $L^\alpha$-risk, $\alpha<3/2$', r'$\mathbf{E}[\,\overline{m m_n}^{\,\alpha}\,]^{1/\alpha}$')
step = alphas[1] - alphas[0] if len(alphas) > 1 else 0.1
bounds = [a - step / 2 for a in alphas] + [alphas[-1] + step / 2]
sm = plt.cm.ScalarMappable(cmap=ListedColormap([COL[a] for a in alphas]), norm=BoundaryNorm(bounds, len(alphas)))
cb = fig.colorbar(sm, ax=axs, ticks=alphas, fraction=0.035, pad=0.02, aspect=30)
cb.set_label(r'$\alpha$', rotation=0, labelpad=6); cb.ax.tick_params(labelsize=7.5, length=0); cb.outline.set_visible(False)
cb.ax.set_yticklabels([f'{a:g}' for a in alphas])
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200); plt.close(fig)

# Monte Carlo diagnostic: running estimate of the L^alpha risk at the largest n
n0 = ns[-1]
fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
for a in alphas:
    e = res[(a, n0)]; Rr = np.arange(1, len(e) + 1)
    run = (np.cumsum(e ** a) / Rr) ** (1 / a)
    ax.plot(Rr[49:], run[49:], '-', color=COL[a], lw=1.2, label=fr'$\alpha={a:g}$')
style_ax(ax, fr'Running Monte Carlo estimate, $n={n0}$', r'estimate of $\mathbf{E}[\,\overline{m m_n}^{\,\alpha}\,]^{1/\alpha}$')
ax.set_xscale('log'); ax.set_xticks([1e2, 1e3, 1e4, 1e5]); ax.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
ax.set_xlabel('number of replications'); ax.legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
fig.savefig(out + '_mc.pdf'); fig.savefig(out + '_mc.png', dpi=200); plt.close(fig)

with open(out + '_summary.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['alpha', 'replications', f'slope_L1(n>={fitmin})', 'slope_median', 'predicted_slope', 'slope_Lalpha', 'max_iter',
                'not_converged', 'recomputed_high_precision'])
    for a in alphas:
        w.writerow([a, len(res[(a, ns[0])]), f"{slope(tab[a]['L1']):.3f}", f"{slope(tab[a]['med']):.3f}", f'{pred(a):.3f}',
                    f"{slope(tab[a]['La']):.3f}" if a < nu else 'infinite',
                    max(info[(a, n)]['max_iter'] for n in ns), sum(info[(a, n)]['not_converged'] for n in ns),
                    sum(info[(a, n)].get('recomputed_high_precision', 0) for n in ns)])
with open(out + '_risks.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['alpha', 'n', 'replications', 'L1', 'L1_stderr', 'Lalpha_MC(meaningful only for alpha<1.5)', 'median', 'q90', 'max',
                'max_iter', 'not_converged', 'recomputed_high_precision'])
    for r in rows: w.writerow([r[0], r[1], r[2]] + [f'{x:.6g}' for x in r[3:9]] + [r[9], r[10], r[11]])
print(open(out + '_summary.csv').read())
