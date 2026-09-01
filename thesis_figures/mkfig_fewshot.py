"""Figure, one external infarction beat under three protocols: zero-shot, label-supervised
few-shot adaptation, and the distilled few-shot adaptation.

Faithful to the original two-panel mkfig_fewshot.py, updated for the v2 model: new exemplar
(InferiorInfarction_008, lead V2), the v2 per-unit results, a third panel for the distilled
adaptation, and the Figure 4.1 wave hues (red QRS, green T).
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch, Rectangle
import figstyle as F
F.apply()

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS = 500.0
WAVES = [('QRS', 'qrs_onset', 'qrs_offset', None, F.RED),
         ('T', 't_onset', 't_offset', 't_peak', F.GREEN)]
QRS_PEAKS = ['q_peak', 'r_peak', 's_peak']
BOUND = ['qrs_onset', 'qrs_offset', 't_onset', 't_offset']

REC, LEAD, UNIT = 'AnteriorInfarction_009', 'aVF', 'simulator units'
CSVDIR = 'ml_modelling/results'
PANELS = [
    (CSVDIR + '/zeroshot_heldout50_2/per_unit.csv',
     '(a) anterior infarction, lead aVF — zero-shot, no adaptation'),
    (CSVDIR + '/fewshot_ext_adapted2/per_unit.csv',
     '(b) anterior infarction, lead aVF — label-supervised adaptation'),
    (CSVDIR + '/fewshot_ext_adapted5/per_unit.csv',
     '(c) anterior infarction, lead aVF — distilled adaptation'),
]
NICE = {'qrs_onset': 'QRS on', 'qrs_offset': 'QRS off', 't_onset': 'T on', 't_offset': 'T off'}

rows = [pd.read_csv(p)[lambda d: (d.record_id == REC) & (d.lead == LEAD)].iloc[0]
        for p, _ in PANELS]
sig = np.load('ml_modelling/data/signal_cache/%s.npy' % REC)[LEADS.index(LEAD)].astype(float)

# one shared window across both panels: reference marks plus every prediction either model makes
pos = []
for r in rows:
    for c in BOUND:
        for suf in ('_pred', '_true'):
            v = r[c + suf]
            if np.isfinite(v):
                pos.append(float(v))
lo = max(0, int(min(pos)) - 55)
hi = min(len(sig), int(max(pos)) + 60)
seg = sig[lo:hi]
span = (hi - lo) / FS * 1000.0
vmin, vmax = float(seg.min()), float(seg.max())
r0 = max(vmax - vmin, 1e-9)

fig, axes = plt.subplots(3, 1, figsize=(F.TEXTWIDTH, 8.0), gridspec_kw={'hspace': 0.70})

for ax, row, (_, title) in zip(axes, rows, PANELS):
    band_hi = vmax + 0.10 * r0
    strip_hi, strip_lo = vmin - 0.16 * r0, vmin - 0.25 * r0
    err_y = vmin - 0.42 * r0
    t = (np.arange(lo, hi) - lo) / FS * 1000.0

    def x(v):
        return np.nan if not np.isfinite(v) else (float(v) - lo) / FS * 1000.0

    for name, a, b, _pk, col in WAVES:
        va, vb = row[a + '_true'], row[b + '_true']
        if np.isfinite(va) and np.isfinite(vb):
            ax.add_patch(Rectangle((x(va), vmin - 0.06 * r0), x(vb) - x(va),
                                   band_hi - (vmin - 0.06 * r0), facecolor=col, alpha=0.15,
                                   edgecolor='none', zorder=1))
            ax.text((x(va) + x(vb)) / 2, band_hi + 0.03 * r0, name, ha='center', va='bottom',
                    fontsize=7.5, color=col, zorder=6)

    for name, a, b, _pk, col in WAVES:
        for m in (a, b):
            v = row[m + '_true']
            if np.isfinite(v) and 0 <= x(v) <= span:
                ax.plot([x(v)] * 2, [vmin - 0.06 * r0, band_hi], color=F.INK, lw=0.9,
                        alpha=0.65, zorder=4)

    for name, a, b, _pk, col in WAVES:
        for m in (a, b):
            v = row[m + '_pred']
            if np.isfinite(v) and 0 <= x(v) <= span:
                ax.plot([x(v)] * 2, [strip_lo, band_hi + 0.02 * r0], color=col, lw=0.9,
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

    err = '   '.join('%s %+.0f' % (NICE[c], row[c + '_err_ms']) for c in BOUND)
    ax.text(0, err_y, 'signed error, ms      ' + err, ha='left', va='center',
            fontsize=6.9, color=F.MUTED, zorder=8)

    ax.set_title(title, loc='left', fontsize=8.6, color=F.INK, pad=10)
    ax.set_xlim(0, span)
    ax.set_ylim(vmin - 0.54 * r0, band_hi + 0.16 * r0)
    ticks = [v for v in MaxNLocator(4).tick_values(vmin, vmax) if vmin <= v <= vmax]
    ax.set_yticks(ticks)
    ax.set_ylabel('lead %s (%s)' % (LEAD, UNIT), labelpad=2)
    ax.set_xlabel('time within the window (ms)', labelpad=1)
    F.despine(ax, keep=('left', 'bottom'))
    ax.spines['left'].set_bounds(vmin, vmax)
    ax.spines['bottom'].set_position(('outward', 4))

from matplotlib.lines import Line2D
handles = [Patch(facecolor=F.WAVE_BG, label='background'),
           Patch(facecolor=F.RED, label='QRS'),
           Patch(facecolor=F.GREEN, label='T'),
           Line2D([], [], color=F.INK, ls='-', lw=0.9, alpha=0.65, label='reference boundary'),
           Line2D([], [], color=F.MUTED, ls='--', dashes=(4, 2.6), lw=0.9,
                  label='predicted boundary')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.045),
           ncol=5, borderaxespad=0, handlelength=1.5, handleheight=0.85, columnspacing=1.6)
fig.savefig('Dissertation/images/fig_fewshot_examples.pdf')
fig.savefig('Dissertation/images/fig_fewshot_examples.png', dpi=400)
for r_, (_, ttl) in zip(rows, PANELS):
    print('%-58s T on err %+6.0f | T off err %+7.0f' % (ttl[:58], r_['t_onset_err_ms'], r_['t_offset_err_ms']))
print('reference J->T onset %.0f ms' % ((rows[0]['t_onset_true'] - rows[0]['qrs_offset_true']) * 2))
print('written both panels, record %s lead %s, span %.0f ms' % (REC, LEAD, span))
