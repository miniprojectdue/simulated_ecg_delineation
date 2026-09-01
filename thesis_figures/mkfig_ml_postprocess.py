"""Figure fig_ml_postprocess (Chapter 3): the staged decoding of one unit, from the class
posteriors to the final boundaries.

Five panels: (a) the per-sample posteriors, (b) the raw argmax labels, (c) after the mode
filter, (d) after the minimum-region and largest-region rules, (e) the boundaries and peaks
read from the cleaned labels, with the QRS-onset isoelectric reference and the deflection
naming threshold. Runs the checkpoint on one unit through the same dataset pipeline as
evaluate.py. Run from the repository root:
"""

import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.getcwd(), 'ml_modelling', 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F
F.apply()

import torch
from evaluate import load_checkpoint
from dataset import BeatWindowDataset, dataset_kwargs, collate

# ---------------------------------------------------------------- EDIT HERE
CHECKPOINT = 'ml_modelling/checkpoints/finetune_toffset_fix_tailweight/best_geometry.pt'
UNITS_CSV = 'ml_modelling/data/finetune_units.csv'
OUT_PNG = 'Dissertation/images/fig_ml_postprocess.png'
OUT_PDF = 'Dissertation/images/fig_ml_postprocess.pdf'
RECORD = None      # None picks the first sinus unit on LEAD with a P wave present
LEAD = 'II'
MODE_KERNEL = 9    # samples, the mode (median-of-labels) filter width
MIN_REGION = 8     # samples, regions shorter than this are discarded
PAD_MS = 120       # display margin either side of the supervised window
CLASSES = [('background', 0, '0.62'), ('P', 1, F.WAVE_P), ('QRS', 2, F.WAVE_QRS),
           ('T', 3, F.WAVE_T)]
STRIP = {0: '0.85', 1: F.WAVE_P, 2: F.WAVE_QRS, 3: F.WAVE_T}
# ---------------------------------------------------------------------------

FS = 500.0
device = torch.device('cpu')
model, cfg, _ = load_checkpoint(CHECKPOINT, device)

d = pd.read_csv(UNITS_CSV)
CACHE_DIR = 'ml_modelling/data/signal_cache'
sel = d[(d.lead == LEAD) & (d.p_present == 1)]
sel = sel[sel.record_id.map(lambda r: os.path.exists('%s/%s.npy' % (CACHE_DIR, r)))]
if sel.empty:
    raise SystemExit('no eligible unit has a cached signal under %s' % CACHE_DIR)
if RECORD is not None:
    sel = sel[sel.record_id == RECORD]
elif (sel.disease_class == 'sinus').any():
    sel = sel[sel.disease_class == 'sinus']
frame = sel.iloc[[0]].reset_index(drop=True)
rec = frame.record_id.iloc[0]

fs_hz = float(frame['fs_hz'].iloc[0]) if 'fs_hz' in frame.columns else FS
dataset = BeatWindowDataset(frame, augment=None, training=False,
                            **dataset_kwargs(cfg, training=False, seed=0, fs_hz=fs_hz))
batch = collate([dataset[0]])
with torch.no_grad():
    out = model(batch['signal'], batch['lead_idx'],
                valid=batch.get('valid'))
logits = (out[0] if isinstance(out, tuple) else out)
post = torch.softmax(logits, dim=1)[0].numpy()          # (4, crop)
trace = batch['trace'][0].numpy()
supervised = batch['supervised'][0].numpy().astype(bool)
target = batch['target'][0].numpy()          # the reference label vector, crop-aligned

# display window: the supervised region plus a margin. Everything downstream is cropped to
# this window first, so the largest-region rule cannot be captured by a neighbouring beat.
idx = np.where(supervised)[0]
pad = int(PAD_MS / 1000.0 * FS)
lo = max(0, int(idx.min()) - pad)
hi = min(post.shape[1], int(idx.max()) + pad)
post = post[:, lo:hi]
trace = trace[lo:hi]
supervised = supervised[lo:hi]
target = target[lo:hi]
t_ms = np.arange(hi - lo) / FS * 1000.0

# the staged decode
labels_raw = post.argmax(axis=0)

def mode_filter(labels, kernel):
    half = kernel // 2
    out = labels.copy()
    for i in range(len(labels)):
        w = labels[max(0, i - half): i + half + 1]
        out[i] = np.bincount(w).argmax()
    return out

labels_mode = mode_filter(labels_raw, MODE_KERNEL)

def clean(labels):
    out = np.zeros_like(labels)
    for _, cid, _c in CLASSES[1:]:
        m = labels == cid
        edges = np.where(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))[0].reshape(-1, 2)
        edges = [e for e in edges if e[1] - e[0] >= MIN_REGION]
        if edges:
            s0, s1 = max(edges, key=lambda e: e[1] - e[0])
            out[s0:s1] = cid
    return out

labels_clean = clean(labels_mode)

# reference regions, read from the re-derived annotation vector rather than the model
def regions(labels, cid):
    m = labels == cid
    if not m.any():
        return []
    e = np.where(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))[0].reshape(-1, 2)
    return [(int(a), int(min(b, len(labels) - 1))) for a, b in e]

REF = {cid: regions(target, cid) for _, cid, _c in CLASSES[1:]}

fig = plt.figure(figsize=(F.TEXTWIDTH * 1.7, 8.0))
gs = fig.add_gridspec(5, 1, height_ratios=[1.7, 0.42, 0.42, 0.42, 1.7], hspace=0.85)

# (a) posteriors
ax = fig.add_subplot(gs[0])
for name, cid, col in CLASSES:
    ax.plot(t_ms, post[cid], color=col, lw=1.4, label=name)
ax.set_xlim(0, t_ms[-1])
ax.set_ylim(-0.03, 1.28)
ax.set_ylabel('$p_\\theta(y_t = c \\mid x)$')
ax.set_title('(a)  the per-sample class posteriors returned by the softmax over the four logits',
             loc='left', fontsize=10)
ax.legend(ncol=4, loc='upper center', frameon=False, fontsize=9)
F.despine(ax)

def strip(ax, labels, title):
    for cid, col in STRIP.items():
        m = labels == cid
        if m.any():
            edges = np.where(np.diff(np.concatenate(([0], m.view(np.int8), [0]))))[0].reshape(-1, 2)
            for s0, s1 in edges:
                ax.axvspan(t_ms[s0], t_ms[min(s1, len(t_ms) - 1)], color=col, lw=0)
    ax.set_xlim(0, t_ms[-1])
    ax.set_yticks([])
    ax.set_title(title, loc='left', fontsize=10)
    F.despine(ax, keep=('bottom',))

strip(fig.add_subplot(gs[1]), labels_raw,
      '(b)  $\\hat{y}_t = \\arg\\max_c\\, p_\\theta(y_t = c \\mid x)$, fragmented at the boundaries')
strip(fig.add_subplot(gs[2]), labels_mode,
      '(c)  after the mode filter of width %d samples (%.0f ms)' % (MODE_KERNEL, MODE_KERNEL / FS * 1000.0))
strip(fig.add_subplot(gs[3]), labels_clean,
      '(d)  after discarding regions shorter than %d samples and keeping the largest region per class' % MIN_REGION)

# (e) boundaries and peaks on the trace
ax = fig.add_subplot(gs[4])
tr = trace
vmin, vmax = float(tr.min()), float(tr.max())
r0 = max(vmax - vmin, 1e-9)
qrs_on = None
baseline = float(np.median(trace[supervised]))
for name, cid, _c in CLASSES[1:]:
    m = labels_clean == cid
    if not m.any():
        continue
    s0 = int(np.argmax(m))
    s1 = int(len(m) - np.argmax(m[::-1]))
    col = STRIP[cid]
    ax.axvspan(t_ms[s0], t_ms[min(s1, len(t_ms) - 1)], color=col, alpha=0.22, zorder=1)
    for edge in (s0, min(s1, len(t_ms) - 1)):
        ax.axvline(t_ms[edge], color=col, ls='--', lw=1.3, zorder=4)
    seg = trace[s0:s1]
    pk = s0 + int(np.argmax(np.abs(seg - baseline)))
    ax.plot(t_ms[pk], trace[pk], 'o', ms=6, color=col, mec='white', mew=0.8, zorder=6)
    if cid == 2:
        qrs_on = s0
if qrs_on is not None:
    v_on = trace[qrs_on]
    dev = np.abs(trace[labels_clean == 2] - v_on)
    thr = max(0.02, 0.05 * float(dev.max()))
    ax.axhline(v_on, color=F.MUTED, ls=':', lw=1.0, zorder=2)
    ax.axhspan(v_on - thr, v_on + thr, color='0.80', alpha=0.5, zorder=0)
    ax.text(0.02, 0.855, 'the QRS onset voltage is the isoelectric reference and the shaded band '
            'is the naming threshold $\\max(0.02\\,\\mathrm{mV},\\ 0.05\\max|v - v_{\\mathrm{on}}|)$',
            transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color=F.MUTED)
# reference boundaries from the re-derived labels: solid, ink, with a caret on the axis, so
# they are told apart from the dashed predictions by line style as well as by colour
ref_t_on = None
for _, cid, _c in CLASSES[1:]:
    for a, b in REF[cid]:
        for edge in (a, b):
            ax.axvline(t_ms[edge], color=F.INK, ls='-', lw=1.0, alpha=0.70, zorder=3)
            ax.plot(t_ms[edge], vmax + 0.40 * r0, marker='v', ms=4.0, color=F.INK,
                    alpha=0.75, clip_on=False, zorder=7)
    if cid == 3 and REF[cid]:
        ref_t_on = REF[cid][0][0]

pred_t_on = int(np.argmax(labels_clean == 3)) if (labels_clean == 3).any() else None
if ref_t_on is not None and pred_t_on is not None:
    y_a = vmin + 0.16 * r0
    ax.annotate('', xy=(t_ms[ref_t_on], y_a), xytext=(t_ms[pred_t_on], y_a),
                arrowprops=dict(arrowstyle='<|-|>', color=F.INK, lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=9), zorder=8)
    ax.text(0.5 * (t_ms[ref_t_on] + t_ms[pred_t_on]), y_a + 0.035 * r0,
            '%.0f ms' % ((ref_t_on - pred_t_on) / FS * 1000.0),
            ha='center', va='bottom', fontsize=8.5, color=F.INK, zorder=8)

ax.plot(t_ms, tr, color=F.INK, lw=1.6, zorder=5)
ax.set_xlim(0, t_ms[-1])
ax.set_ylim(vmin - 0.12 * r0, vmax + 0.46 * r0)
ax.set_ylabel('lead %s (mV)' % LEAD)
ax.set_xlabel('time within the window (ms)')
ax.set_title('(e)  onsets and offsets read from the label transitions, peaks read from the '
             'voltage trace inside each region', loc='left', fontsize=10)
F.despine(ax)

from matplotlib.lines import Line2D
handles = [Patch(facecolor=STRIP[0], label='background'),
           Patch(facecolor=STRIP[1], label='P'),
           Patch(facecolor=STRIP[2], label='QRS'),
           Patch(facecolor=STRIP[3], label='T'),
           Line2D([], [], color=F.MUTED, ls='--', lw=1.3, label='predicted boundary'),
           Line2D([], [], color=F.INK, ls='-', lw=1.0, alpha=0.70,
                  label='reference boundary')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.012), ncol=6,
           borderaxespad=0, handlelength=1.5, handleheight=0.9, columnspacing=1.8)
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print('unit %s lead %s, display samples %d..%d of the crop' % (rec, LEAD, lo, hi))
if ref_t_on is not None and pred_t_on is not None:
    qrs_off = REF[2][0][1] if REF[2] else None
    print('T onset  predicted %.0f ms | reference %.0f ms | gap %.0f ms'
          % (pred_t_on / FS * 1000.0, ref_t_on / FS * 1000.0,
             (ref_t_on - pred_t_on) / FS * 1000.0))
    if qrs_off is not None:
        print('J point  %.0f ms | predicted T onset is J%+.0f ms | reference T onset is J%+.0f ms'
              % (qrs_off / FS * 1000.0, (pred_t_on - qrs_off) / FS * 1000.0,
                 (ref_t_on - qrs_off) / FS * 1000.0))
