"""Figure 2.x - Main components of a normal ECG waveform.

Vector redraw of the schematic previously included as a low-resolution raster
(images/ecg_waveform_components.png).  Every label, bracket, numeric value and
callout of the source figure is preserved. only the rendering is changed, so the
figure stays sharp at any print size.

Outputs: images/fig_ecg_waveform_components.pdf  (vector, used by main.tex)
         images/fig_ecg_waveform_components.png  (400 dpi raster fallback)
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ----------------------------------------------------------------- style ----
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'figure.facecolor': 'white',
})

TRACE  = '#453E93'   # ECG trace (sampled from the source figure)
WAVE   = '#8C3A3A'   # P Q R S T U letters
INK    = '#1F2225'   # primary label text
ACCENT = '#2F6091'   # the "(distance between ...)" parentheticals
GREY   = '#4E5257'   # secondary / explanatory text
RULE   = '#3A3D40'   # bracket rules
BOXBG  = '#E7E8EA'
BAND   = '#D2D5D9'

FS_BIG   = 9.6   # P Q R S T U
FS_LAB   = 8.0    # bracket labels
FS_SMALL = 6.9    # explanatory text
LEAD     = 0.168  # vertical line spacing for stacked text, in data units

TEXTWIDTH = 5.9   # inches (a4paper, 3 cm margins)

# ------------------------------------------------------------- geometry ----
# Horizontal layout, arbitrary units; one beat repeats with period DT.
DT = 62.0

P1, R1 = 12.0, 21.8
P_HW   = 3.6                      # half-width of the P wave footprint
Q_ON   = R1 - 1.8                 # QRS onset
J      = R1 + 2.1                 # J point (QRS offset)
T1     = 38.0
T_ON, T_OFF = 30.0, 44.0
U1     = 52.0

P2, R2 = P1 + DT, R1 + DT
Q_ON2  = Q_ON + DT

Y_R, Y_Q, Y_S = 1.30, -0.13, -0.32
Y_P, Y_T, Y_U = 0.18, 0.36, 0.105


def gauss(x, c, w, a):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def qrs(x, r):
    """Piecewise-linear QRS complex centred on R peak `r`."""
    pts = [(r - 1.8, 0.0), (r - 0.9, Y_Q), (r, Y_R), (r + 1.1, Y_S), (r + 2.1, 0.0)]
    px, py = zip(*pts)
    return np.where((x >= px[0]) & (x <= px[-1]), np.interp(x, px, py), 0.0)


x = np.linspace(-2, 103, 12000)
y = (gauss(x, P1, 1.55, Y_P) + gauss(x, P2, 1.55, Y_P)
     + gauss(x, T1, 3.4, Y_T)
     + qrs(x, R1) + qrs(x, R2))

xu = np.linspace(U1 - 5.0, U1 + 5.0, 400)
yu = gauss(xu, U1, 2.2, Y_U)

# ---------------------------------------------------------------- canvas ----
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.62))
ax.set_xlim(-2.0, 103.0)
ax.set_ylim(-2.26, 2.30)
ax.axis('off')

# reference band dropping from the beat-2 PR segment to the note box
BAND_X0, BAND_X1 = 78.1, 80.2
BOX_TOP, BOX_BOT = -0.74, -1.74
ax.add_patch(Rectangle((BAND_X0, BOX_TOP), BAND_X1 - BAND_X0, 0.02 - BOX_TOP,
                       facecolor=BAND, edgecolor='none', zorder=0))

ax.plot(x, y, color=TRACE, lw=1.75, solid_joinstyle='round',
        solid_capstyle='round', zorder=3)
ax.plot(xu, yu, color=TRACE, lw=1.6, ls=(0, (1.1, 1.4)), zorder=3)


# ------------------------------------------------------------- helpers -----
def bracket(x0, x1, y, tip=0.10, down=True, lw=0.85):
    """Square bracket: rule at `y` with end ticks pointing at the trace."""
    s = -tip if down else tip
    ax.plot([x0, x1], [y, y], color=RULE, lw=lw, solid_capstyle='butt', zorder=4)
    ax.plot([x0, x0], [y, y + s], color=RULE, lw=lw, solid_capstyle='butt', zorder=4)
    ax.plot([x1, x1], [y, y + s], color=RULE, lw=lw, solid_capstyle='butt', zorder=4)


def label(xc, y, s, fs=FS_LAB, color=INK, ha='center', weight='normal'):
    return ax.text(xc, y, s, fontsize=fs, color=color, ha=ha, va='center',
                   weight=weight, zorder=5)


_runs = []


def run(xc, y, parts, fs=FS_LAB, align='center', gap=0.9):
    """A single line built from (text, colour, weight) parts, laid out after a
    first render pass so mixed-colour lines centre correctly."""
    handles = [ax.text(0, y, s, fontsize=fs, color=c, weight=w, ha='left',
                       va='center', zorder=5) for s, c, w in parts]
    _runs.append((xc, handles, align, gap))
    return handles


def place_runs():
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for xc, handles, align, gap in _runs:
        widths = []
        for h in handles:
            bb = h.get_window_extent(fig.canvas.get_renderer())
            (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            widths.append(x1 - x0)
        total = sum(widths) + gap * (len(handles) - 1)
        cx = xc - total / 2 if align == 'center' else xc
        for h, w in zip(handles, widths):
            h.set_x(cx)
            cx += w + gap


# ------------------------------------------------------- above the trace ----
bracket(R1, R2, 2.02)
run((R1 + R2) / 2, 2.14, [('RR interval', INK, 'normal'),
                          ('(distance between R-waves)', ACCENT, 'normal')])

bracket(P1, P2, 1.66)
run((P1 + P2) / 2, 1.78, [('PP interval', INK, 'normal'),
                          ('(distance between P-waves)', ACCENT, 'normal')])

bracket(J, T_OFF, 1.04)
label((J + T_OFF) / 2, 1.16, 'ST-T segment')

bracket(T_OFF, P2 - P_HW, 1.04)
label((T_OFF + P2 - P_HW) / 2, 1.16, 'TP interval')

# PR segment (P offset -> QRS onset)
_prc = (P1 + P_HW + Q_ON) / 2 - 2.0
bracket(P1 + P_HW, Q_ON, 0.58, tip=0.09)
label(_prc, 0.88 + LEAD, 'PR')
label(_prc, 0.88, 'segment')

# P duration
bracket(P1 - P_HW, P1 + P_HW, 0.40, tip=0.09)
label(P1 - 1.0, 0.53, 'P duration')

# ST segment (J point -> T onset)
bracket(J, T_ON, 0.40, tip=0.09)
label(J - 0.1, 0.53, 'ST segment', ha='left')

# ------------------------------------------------------------- wave names ---
for xx, yy, s in [(P1, 0.36, 'P'), (Q_ON + 0.2, -0.50, 'Q'), (R1, 1.46, 'R'),
                  (R1 + 2.5, -0.50, 'S'), (T1, 0.60, 'T'), (U1, 0.31, 'U'),
                  (P2, 0.36, 'P'), (Q_ON2 + 0.2, -0.50, 'Q'), (R2, 1.46, 'R'),
                  (R2 + 2.5, -0.50, 'S')]:
    label(xx, yy, s, fs=FS_BIG, color=WAVE)

# ------------------------------------------------------- below the trace ----
_prl = (P1 - P_HW + Q_ON) / 2 - 2.3
bracket(P1 - P_HW, Q_ON, -0.30, down=False)
label(_prl, -0.60, 'PR interval')
label(_prl, -0.60 - LEAD, '0.12 – 0.22 s')

bracket(Q_ON, J, -0.97, down=False)
label((Q_ON + J) / 2, -1.15, 'QRS duration')
label((Q_ON + J) / 2, -1.15 - LEAD, '< 0.12 s')

bracket(Q_ON, T_OFF, -1.58, down=False)
label((Q_ON + T_OFF) / 2, -1.76, 'QT duration')
label((Q_ON + T_OFF) / 2, -1.76 - LEAD, 'Corrected QT duration for men: ≤ 0.45 s', fs=FS_SMALL)
label((Q_ON + T_OFF) / 2, -1.76 - 2 * LEAD, 'Corrected QT duration for women: ≤ 0.47 s', fs=FS_SMALL)

# ------------------------------------------------------ J point callouts ----
JX, J60X = J + 0.55, J + 2.6
for xx in (JX, J60X):
    ax.plot([xx], [0.02], marker='o', ms=4.2, mfc='white', mec=INK,
            mew=0.8, zorder=6)
    ax.plot([xx], [0.02], marker='o', ms=1.1, mfc=INK, mec=INK, zorder=7)

TX = 34.0
ax.plot([J60X + 0.5, TX - 1.2], [0.02, -0.30], color=INK, lw=0.75, zorder=4)
ax.plot([JX + 0.4, JX + 4.0, TX - 1.2], [-0.05, -0.66, -1.00],
        color=INK, lw=0.75, zorder=4)

label(TX, -0.30, 'J-60 point', ha='left', weight='bold')
label(TX, -0.30 - LEAD, 'Measurement of ST depression in', ha='left', fs=FS_SMALL, color=GREY)
label(TX, -0.30 - 2 * LEAD, 'exercise stress testing', ha='left', fs=FS_SMALL, color=GREY)

label(TX, -1.00, 'J point', ha='left', weight='bold')
label(TX, -1.00 - LEAD, 'Measurement of ST elevation and', ha='left', fs=FS_SMALL, color=GREY)
label(TX, -1.00 - 2 * LEAD, 'depression in most situations.', ha='left', fs=FS_SMALL, color=GREY)

# ------------------------------------------------------------- note box ----
BX0, BX1 = 72.6, 102.5
ax.add_patch(FancyBboxPatch((BX0, BOX_BOT), BX1 - BX0, BOX_TOP - BOX_BOT,
                            boxstyle='round,pad=0,rounding_size=0.25',
                            facecolor=BOXBG, edgecolor='none', zorder=1))

bx, by = BX0 + 1.6, BOX_TOP - 0.20
run(bx, by, [('The reference level', INK, 'bold'), ('for measuring ST', INK, 'normal')],
    fs=FS_SMALL, align='left', gap=0.55)
for i, line in enumerate(['segment deviation (depression, elevation)',
                          'is the PR segment, which is the baseline',
                          '(isoelectric line) of the ECG.'], start=1):
    label(bx, by - i * LEAD, line, ha='left', fs=FS_SMALL)

# ------------------------------------------------------------------ save ----
place_runs()

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, '..', 'images')
fig.savefig(os.path.join(out, 'fig_ecg_waveform_components.pdf'))
fig.savefig(os.path.join(out, 'fig_ecg_waveform_components.png'), dpi=400)
print('wrote fig_ecg_waveform_components.{pdf,png} to', os.path.normpath(out))
