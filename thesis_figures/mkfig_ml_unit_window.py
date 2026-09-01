"""Figure fig_ml_unit_window (Chapter 3): one training unit as the network sees it.

Three panels: (a) the twelve-lead input crop with the nominated lead in black, (b) the
nominated lead with the reference fiducials and the supervised window, (c) the target vector
with the ignore index hatched. Built from the reviewed units table and the signal cache, so
any unit can be shown by changing RECORD and LEAD below. 

"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch
import figstyle as F
F.apply()

# ---------------------------------------------------------------- EDIT HERE
UNITS_CSV = 'ml_modelling/data/finetune_units.csv'
CACHE_DIR = 'ml_modelling/data/signal_cache'
OUT_PNG = 'Dissertation/images/fig_ml_unit_window.png'
OUT_PDF = 'Dissertation/images/fig_ml_unit_window.pdf'
RECORD = None      # None picks the first sinus unit on LEAD with a P wave present
LEAD = 'II'
CROP = 1280        # samples in the model crop
IGNORE_HATCH = '///'
WAVE = {'P': F.WAVE_P, 'QRS': F.WAVE_QRS, 'T': F.WAVE_T}
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# ---------------------------------------------------------------------------

FS = 500.0
d = pd.read_csv(UNITS_CSV)
sel = d[(d.lead == LEAD) & (d.p_present == 1)]
sel = sel[sel.record_id.map(lambda r: os.path.exists('%s/%s.npy' % (CACHE_DIR, r)))]
if sel.empty:
    raise SystemExit('no eligible unit has a cached signal under %s' % CACHE_DIR)
if RECORD is not None:
    sel = sel[sel.record_id == RECORD]
elif (sel.disease_class == 'sinus').any():
    sel = sel[sel.disease_class == 'sinus']
row = sel.iloc[0]
rec = row.record_id
sig = np.load('%s/%s.npy' % (CACHE_DIR, rec)).astype(float)
n = sig.shape[1]

w0, w1 = int(row.beat_start_sample), int(row.beat_end_sample)
centre = (w0 + w1) // 2
c0 = int(np.clip(centre - CROP // 2, 0, max(0, n - CROP)))
c1 = c0 + CROP
t_ms = (np.arange(c0, c1) - c0) / FS * 1000.0

def x(sample):
    return (float(sample) - c0) / FS * 1000.0

BOUND = [('p_onset_sample', 'P'), ('p_offset_sample', 'P'),
         ('qrs_onset_sample', 'QRS'), ('qrs_offset_sample', 'QRS'),
         ('t_onset_sample', 'T'), ('t_offset_sample', 'T')]
PEAKS = [('p_peak_sample', 'P'), ('q_peak_sample', 'QRS'), ('r_peak_sample', 'QRS'),
         ('s_peak_sample', 'QRS'), ('t_peak_sample', 'T')]
SPANS = [('p_onset_sample', 'p_offset_sample', 'P'),
         ('qrs_onset_sample', 'qrs_offset_sample', 'QRS'),
         ('t_onset_sample', 't_offset_sample', 'T')]

fig = plt.figure(figsize=(F.TEXTWIDTH * 1.7, 8.2))
gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.6, 0.55], hspace=0.55)

# (a) all twelve leads, nominated lead black
ax = fig.add_subplot(gs[0])
lead_idx = LEADS.index(LEAD)
scale = np.percentile(np.abs(sig[:, c0:c1]), 99) * 2.2
for i, name in enumerate(LEADS):
    off = -i
    tr = sig[i, c0:c1] / scale
    ax.plot(t_ms, tr + off, color=(F.INK if i == lead_idx else '0.62'),
            lw=(1.3 if i == lead_idx else 0.8), zorder=(4 if i == lead_idx else 3))
ax.axvspan(x(w0), x(w1), color='0.90', zorder=1)
ax.set_yticks([-i for i in range(12)])
ax.set_yticklabels([('$\\bf{%s}$' % l) if l == LEAD else l for l in LEADS], fontsize=8.5)
ax.set_xlim(0, CROP / FS * 1000.0)
ax.set_xlabel('time within the crop (ms)')
ax.set_title('(a)  the input tensor, all twelve leads over the %d sample crop, with lead %s '
             'nominated for delineation' % (CROP, LEAD), loc='left', fontsize=10)
F.despine(ax, keep=('bottom',))
ax.tick_params(axis='y', length=0)

# (b) the nominated lead with reference fiducials and the supervised window
ax = fig.add_subplot(gs[1])
tr = sig[lead_idx, c0:c1]
vmin, vmax = float(tr.min()), float(tr.max())
r0 = max(vmax - vmin, 1e-9)
ax.axvspan(x(w0), x(w1), color='0.92', zorder=0)
for a, b, w in SPANS:
    va, vb = row[a], row[b]
    if np.isfinite(va) and np.isfinite(vb):
        ax.axvspan(x(va), x(vb), color=WAVE[w], alpha=0.28, zorder=1)
for col, w in BOUND:
    v = row[col]
    if np.isfinite(v):
        ax.axvline(x(v), color=WAVE[w], ls='--', lw=1.1, zorder=4)
        ax.plot(x(v), tr[int(v) - c0], 'o', ms=5, color=WAVE[w], mec='white', mew=0.6, zorder=6)
for col, w in PEAKS:
    v = row[col]
    if np.isfinite(v):
        ax.plot(x(v), tr[int(v) - c0], 'o', ms=6, color=WAVE[w], mec='white', mew=0.8, zorder=6)
ax.plot(t_ms, tr, color=F.INK, lw=1.5, zorder=5)
yarrow = vmax + 0.16 * r0
ax.add_patch(FancyArrowPatch((x(w0), yarrow), (x(w1), yarrow), arrowstyle='<->',
                             mutation_scale=14, color=F.MUTED, lw=1.2))
ax.text((x(w0) + x(w1)) / 2, yarrow + 0.03 * r0,
        'supervised window, %d samples (%.0f ms)' % (w1 - w0, (w1 - w0) / FS * 1000.0),
        ha='center', va='bottom', fontsize=9.5, color=F.MUTED)
ax.text(x(w0) / 2, yarrow + 0.03 * r0, 'context, loss not applied',
        ha='center', va='bottom', fontsize=9.5, color=F.MUTED)
ax.text((x(w1) + CROP / FS * 1000.0) / 2, yarrow + 0.03 * r0, 'context, loss not applied',
        ha='center', va='bottom', fontsize=9.5, color=F.MUTED)
ax.set_xlim(0, CROP / FS * 1000.0)
ax.set_ylim(vmin - 0.12 * r0, yarrow + 0.28 * r0)
ax.set_ylabel('lead %s (mV)' % LEAD)
ax.set_xlabel('time within the crop (ms)')
ax.set_title('(b)  the nominated lead with the reference fiducials present in this unit, '
             'neighbouring beats visible as context', loc='left', fontsize=10)
F.despine(ax)

# (c) the target vector, hatched where the ignore index suppresses the loss
ax = fig.add_subplot(gs[2])
span_ms = CROP / FS * 1000.0
ax.add_patch(Rectangle((0, 0), x(w0), 1, facecolor='white', edgecolor='0.62',
                       hatch=IGNORE_HATCH, lw=0.8, zorder=1))
ax.add_patch(Rectangle((x(w1), 0), span_ms - x(w1), 1, facecolor='white', edgecolor='0.62',
                       hatch=IGNORE_HATCH, lw=0.8, zorder=1))
ax.add_patch(Rectangle((x(w0), 0), x(w1) - x(w0), 1, facecolor='0.85', edgecolor='none', zorder=2))
for a, b, w in SPANS:
    va, vb = row[a], row[b]
    if np.isfinite(va) and np.isfinite(vb):
        ax.add_patch(Rectangle((x(va), 0), x(vb) - x(va), 1, facecolor=WAVE[w],
                               edgecolor='none', zorder=3))
ax.set_xlim(0, span_ms)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel('time within the crop (ms)')
ax.set_title('(c)  the target vector $y \\in \\{-100, 0, 1, 2, 3\\}^{%d}$, hatched where the '
             'ignore index suppresses the loss' % CROP, loc='left', fontsize=10)
F.despine(ax, keep=('bottom',))

handles = [Patch(facecolor='white', edgecolor='0.62', hatch=IGNORE_HATCH, label='ignore index'),
           Patch(facecolor='0.85', label='background'),
           Patch(facecolor=WAVE['P'], label='P'),
           Patch(facecolor=WAVE['QRS'], label='QRS'),
           Patch(facecolor=WAVE['T'], label='T')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.015), ncol=5,
           borderaxespad=0, handlelength=1.5, handleheight=0.9, columnspacing=1.8)
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print('unit %s lead %s, window %d..%d, crop %d..%d' % (rec, LEAD, w0, w1, c0, c1))
