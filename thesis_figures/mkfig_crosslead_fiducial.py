"""Figure fig_crosslead_fiducial (Chapter 4): per-fiducial cross-lead spread and the
QRS+T primary flag rate by lead.

Recomputes everything from the QC output of ecgdeli_labelling/scripts/crosslead_fiducial_qc.py,
so the figure cannot drift from the QC numbers. Run from the repository root:

    python3 Dissertation/ImageCodes/mkfig_crosslead_fiducial.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figstyle as F
F.apply()

# ---------------------------------------------------------------- EDIT HERE
FLAGS_CSV = 'ecgdeli_labelling/data/qc/crosslead_fiducial_flags.csv'
OUT_PNG = 'Dissertation/images/fig_crosslead_fiducial.png'
OUT_PDF = 'Dissertation/images/fig_crosslead_fiducial.pdf'
# fiducials shown in the left panel, top to bottom (Q is omitted as in the document)
FIDS = ['Pon', 'Ppk', 'Poff', 'R', 'S', 'J', 'Ton', 'Tpk', 'Toff']
NICE = {'Pon': 'P onset', 'Ppk': 'P peak', 'Poff': 'P offset', 'R': 'R peak', 'S': 'S peak',
        'J': 'J (QRS off)', 'Ton': 'T onset', 'Tpk': 'T peak', 'Toff': 'T offset'}
COLOUR = {'Pon': F.PURPLE, 'Ppk': F.PURPLE, 'Poff': F.PURPLE,
          'R': F.RED, 'S': F.RED, 'J': F.RED,
          'Ton': F.GREEN, 'Tpk': F.GREEN, 'Toff': F.GREEN}
LEGEND = [('QRS landmark (primary)', F.RED), ('T landmark (primary)', F.GREEN),
          ('P landmark (context)', F.PURPLE)]
# fiducials whose flags count towards the right-hand panel
PRIMARY = ['R', 'S', 'J', 'Ton', 'Tpk', 'Toff']
REF_LEADS = ['V2', 'V5']
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# ---------------------------------------------------------------------------

d = pd.read_csv(FLAGS_CSV)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * F.TEXTWIDTH * 0.95, 4.4),
                               gridspec_kw={'width_ratios': [1.25, 1.0]})

# left panel: dot = median deviation, bar = 98th-percentile tolerance
y = np.arange(len(FIDS))[::-1]
for yi, fid in zip(y, FIDS):
    a = d['dev_' + fid].dropna().to_numpy()
    med, p98 = float(np.median(a)), float(np.percentile(a, 98))
    c = COLOUR[fid]
    ax1.plot([med, p98], [yi, yi], color=c, lw=3.0, solid_capstyle='butt', zorder=2)
    ax1.plot([p98, p98], [yi - 0.13, yi + 0.13], color=c, lw=2.0, zorder=3)
    ax1.plot(med, yi, 'o', ms=9, color=c, zorder=4)
    ax1.annotate('%.0f' % round(p98), (p98, yi), textcoords='offset points',
                 xytext=(10, -3), fontsize=8.5, color=c)
ax1.set_yticks(y)
ax1.set_yticklabels([NICE[f] for f in FIDS])
ax1.set_xlim(0, 205)
ax1.set_xlabel('Cross-lead deviation from %s/%s reference (ms)' % tuple(REF_LEADS))
ax1.set_title('Per-fiducial spread  (dot = median, bar = 98th-pct tolerance)', fontsize=9.5)
ax1.xaxis.grid(True, zorder=0); ax1.set_axisbelow(True)
handles = [plt.Line2D([0], [0], marker='o', color=c, lw=2.5, label=t) for t, c in LEGEND]
ax1.legend(handles=handles, loc='center right', fontsize=8, frameon=True, framealpha=1.0)
F.despine(ax1)

# right panel: per-lead percentage of units flagged on any QRS+T primary fiducial
flag_any = d[['flag_' + f for f in PRIMARY]].max(axis=1)
rate = (d.assign(_f=flag_any).groupby('lead')['_f'].mean() * 100.0)
order = rate.sort_values(ascending=False).index.tolist()
overall = float(flag_any.mean() * 100.0)
cols = [F.RED if l in REF_LEADS else F.BLUE for l in order]
ax2.bar(np.arange(len(order)), rate[order].to_numpy(), color=cols, width=0.72, zorder=3)
ax2.axhline(overall, color=F.MUTED, ls='--', lw=1.2, zorder=2)
ax2.annotate('overall %.1f%%' % overall, (len(order) - 0.4, overall),
             textcoords='offset points', xytext=(0, 5), ha='right', fontsize=8.5, color=F.MUTED)
ax2.set_xticks(np.arange(len(order)))
ax2.set_xticklabels(order)
ax2.set_xlabel('Lead')
ax2.set_ylabel('Units flagged (%)')
ax2.set_title('QRS+T primary flag rate by lead', fontsize=9.5)
ax2.yaxis.grid(True, zorder=0); ax2.set_axisbelow(True)
F.despine(ax2)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_PDF)
print('overall flag rate %.2f%%, worst lead %s %.1f%%' % (overall, order[0], rate[order[0]]))
