
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch, Rectangle
import figstyle as F
F.apply()

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS = 500.0
WAVES = [('P', 'p_onset', 'p_offset', 'p_peak', F.WAVE_P),
         ('QRS', 'qrs_onset', 'qrs_offset', None, F.WAVE_QRS),
         ('T', 't_onset', 't_offset', 't_peak', F.WAVE_T)]
QRS_PEAKS = ['q_peak', 'r_peak', 's_peak']

PANELS = [
    ('ml_modelling/results/indist_toffsetfix/per_unit.csv',
     'mc_lbbb_train_run_S70_run_000012', 'V3', 'mV',
     '(a) in-distribution, left bundle branch block, lead V3',
     ['p_onset', 'p_offset', 'qrs_onset', 'qrs_offset', 't_onset', 't_offset']),
    ('ml_modelling/results/zeroshot_ext_qrs_off/per_unit.csv',
     'AnteriorIschemia_018', 'II', 'simulator units',
     '(b) external, anterior ischemia, lead II',
     ['qrs_onset', 'qrs_offset', 't_onset', 't_offset']),
]
NICE = {'p_onset': 'P on', 'p_offset': 'P off', 'qrs_onset': 'QRS on',
        'qrs_offset': 'QRS off', 't_onset': 'T on', 't_offset': 'T off'}

fig, axes = plt.subplots(2, 1, figsize=(F.TEXTWIDTH, 5.4),
                         gridspec_kw={'hspace': 0.70})

for ax, (path, rec, lead, unit, title, boundaries) in zip(axes, PANELS):
    d = pd.read_csv(path)
    row = d[(d.record_id == rec) & (d.lead == lead)].iloc[0]
    sig = np.load('ml_modelling/data/signal_cache/%s.npy' % rec)[LEADS.index(lead)].astype(float)

    pos = [float(row[c + '_pred']) for c in boundaries if np.isfinite(row[c + '_pred'])]
    pos += [float(row[c + '_true']) for c in boundaries if np.isfinite(row[c + '_true'])]
    lo = max(0, int(min(pos)) - 55)
    hi = min(len(sig), int(max(pos)) + 60)
    seg = sig[lo:hi]                                   # left in the corpus's own units
    t = (np.arange(lo, hi) - lo) / FS * 1000.0
    span = (hi - lo) / FS * 1000.0

    vmin, vmax = float(seg.min()), float(seg.max())
    r = max(vmax - vmin, 1e-9)
    band_hi = vmax + 0.10 * r                          # top of the shaded reference band
    strip_hi, strip_lo = vmin - 0.16 * r, vmin - 0.25 * r
    err_y = vmin - 0.42 * r

    def x(v):
        return np.nan if not np.isfinite(v) else (float(v) - lo) / FS * 1000.0

    for name, a, b, _pk, col in WAVES:
        va, vb = row[a + '_true'], row[b + '_true']
        if np.isfinite(va) and np.isfinite(vb):
            ax.add_patch(Rectangle((x(va), vmin - 0.06 * r), x(vb) - x(va),
                                   band_hi - (vmin - 0.06 * r), facecolor=col, alpha=0.15,
                                   edgecolor='none', zorder=1))
            ax.text((x(va) + x(vb)) / 2, band_hi + 0.03 * r, name, ha='center', va='bottom',
                    fontsize=7.5, color=col, zorder=6)

    for name, a, b, _pk, col in WAVES:
        for m in (a, b):
            v = row[m + '_pred']
            if np.isfinite(v) and 0 <= x(v) <= span:
                ax.plot([x(v)] * 2, [strip_lo, band_hi + 0.02 * r], color=col, lw=0.9,
                        ls='--', dashes=(4, 2.6), zorder=5)

    ax.plot(t, seg, color=F.INK, lw=1.05, zorder=4, solid_joinstyle='round')

    for name, a, b, pk, col in WAVES:
        for m in ([pk] if pk else QRS_PEAKS):
            c = m + '_pred'
            v = float(row[c]) if c in row.index and pd.notna(row[c]) else np.nan
            if np.isfinite(v) and lo <= v < hi:
                ax.plot([x(v)], [seg[int(round(v)) - lo]], 'o', ms=5.0, color=col,
                        mec='white', mew=0.8, zorder=9)

    ax.add_patch(Rectangle((0, strip_lo), span, strip_hi - strip_lo,
                           facecolor=F.WAVE_BG, edgecolor='none', zorder=2))
    for name, a, b, _pk, col in WAVES:
        va, vb = row[a + '_pred'], row[b + '_pred']
        if np.isfinite(va) and np.isfinite(vb):
            xa, xb = max(0, x(va)), min(span, x(vb))
            if xb > xa:
                ax.add_patch(Rectangle((xa, strip_lo), xb - xa, strip_hi - strip_lo,
                                       facecolor=col, edgecolor='none', zorder=3))

    err = '   '.join('%s %+.0f' % (NICE[c], row[c + '_err_ms']) for c in boundaries)
    ax.text(0, err_y, 'signed error, ms      ' + err, ha='left', va='center',
            fontsize=6.9, color=F.MUTED, zorder=8)

    ax.set_title(title, loc='left', fontsize=8.6, color=F.INK, pad=10)
    ax.set_xlim(0, span)
    ax.set_ylim(vmin - 0.54 * r, band_hi + 0.16 * r)
    # ticks confined to the measured range, so none can fall level with the strip below it
    ticks = [v for v in MaxNLocator(4).tick_values(vmin, vmax) if vmin <= v <= vmax]
    ax.set_yticks(ticks)
    ax.set_ylabel('lead %s (%s)' % (lead, unit), labelpad=2)
    ax.set_xlabel('time within the window (ms)', labelpad=1)
    F.despine(ax, keep=('left', 'bottom'))
    ax.spines['left'].set_bounds(vmin, vmax)
    ax.spines['bottom'].set_position(('outward', 4))

handles = [Patch(facecolor=F.WAVE_BG, label='background'),
           Patch(facecolor=F.WAVE_P, label='P'),
           Patch(facecolor=F.WAVE_QRS, label='QRS'),
           Patch(facecolor=F.WAVE_T, label='T')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.035),
           ncol=4, borderaxespad=0, handlelength=1.5, handleheight=0.85, columnspacing=1.6)
fig.savefig('Dissertation/images/fig_examples.pdf')
fig.savefig('Dissertation/images/fig_examples.png', dpi=400)
print('written')
