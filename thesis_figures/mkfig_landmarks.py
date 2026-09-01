"""Figure, landmark localisation error on both test surfaces (v2 model).

Faithful to ml_modelling/figures/mkfig_landmarks.py, updated to read the v2 per-unit results
and with the legend moved below the axis. The x-limit follows the v2 error range.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import figstyle as F
F.apply()

ORDER = ['p_onset', 'p_peak', 'p_offset', 'qrs_onset', 'q_peak', 'r_peak',
         's_peak', 'qrs_offset', 't_onset', 't_peak', 't_offset']
NICE = {'p_onset': 'P onset', 'p_peak': 'P peak', 'p_offset': 'P offset',
        'qrs_onset': 'QRS onset', 'q_peak': 'Q peak', 'r_peak': 'R peak',
        's_peak': 'S peak', 'qrs_offset': 'QRS offset', 't_onset': 'T onset',
        't_peak': 'T peak', 't_offset': 'T offset'}
BOUND = {'p_onset', 'p_offset', 'qrs_onset', 'qrs_offset', 't_onset', 't_offset'}

def mae(path):
    d = pd.read_csv(path)
    return {m: float(np.nanmean(np.abs(d[m + '_err_ms']))) if d[m + '_err_ms'].notna().any()
            else np.nan for m in ORDER}

CSVDIR = 'ml_modelling/results'
ind = mae(CSVDIR + '/indist_toffsetfix2_tailweight/per_unit.csv')
ext = mae(CSVDIR + '/zeroshot_ext_qrs_off2/per_unit.csv')

fig, ax = plt.subplots(figsize=(F.TEXTWIDTH, 3.55))
y = np.arange(len(ORDER))[::-1]
h = 0.36
ax.barh(y + h / 2, [ind[m] for m in ORDER], height=h, color=F.BLUE,
        label='in-distribution, 109 units', zorder=3)
ax.barh(y - h / 2, [0 if np.isnan(ext[m]) else ext[m] for m in ORDER], height=h,
        color=F.CLAY, label='external, 1,200 units', zorder=3)

for i, m in enumerate(ORDER):
    yy = y[i]
    ax.text(ind[m] + 0.3, yy + h / 2, '%.1f' % ind[m], va='center', fontsize=7.2, color=F.INK)
    if np.isnan(ext[m]):
        ax.text(12.5, yy - h / 2, 'not defined in the external reference',
                va='center', fontsize=7.2, color=F.MUTED, style='italic')
    else:
        ax.text(ext[m] + 0.3, yy - h / 2, '%.1f' % ext[m], va='center', fontsize=7.2, color=F.INK)

ax.set_yticks(y)
ax.set_yticklabels([NICE[m] + ('' if m in BOUND else '  ') for m in ORDER],
                   fontweight='normal')
for t, m in zip(ax.get_yticklabels(), ORDER):
    t.set_color(F.INK if m in BOUND else F.MUTED)
ax.set_xlabel('mean absolute error, milliseconds')
ax.set_xlim(0, max(v for v in ext.values() if v == v) * 1.14)
ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
F.despine(ax, keep=('bottom',))
ax.tick_params(axis='y', length=0)
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.045), ncol=2, borderaxespad=0)
fig.savefig('Dissertation/images/fig_landmark_surfaces.pdf')
fig.savefig('Dissertation/images/fig_landmark_surfaces.png', dpi=400)
print({k: round(v, 2) if v == v else None for k, v in ind.items()})
print({k: round(v, 2) if v == v else None for k, v in ext.items()})
