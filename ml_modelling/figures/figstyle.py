
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE   = '#3B6EA5'   # general series, in-distribution surface
RED    = '#C44E52'   # QRS, second surface
GREEN  = '#5FA462'   # T wave
PURPLE = '#7950A4'   # P wave

INK   = '#2F3437'
MUTED = '#6F7780'
GRID  = '#E2E5E8'
BAND  = '#E7E9EC'

CLAY = RED           # kept so older call sites keep working

# The wave colours used by the Chapter 3 pipeline figures, so a wave is the same colour in the
# methods chapter and in the results chapter. These are not the four hues above.
WAVE_BG   = '#D3D3D3'
WAVE_P    = '#3E9A70'
WAVE_QRS  = '#7E72B5'
WAVE_T    = '#E0762D'

DIVERGE = LinearSegmentedColormap.from_list(
    'crosslead_diverge',
    ['#2B527A', '#3B6EA5', '#89A8C6', '#D3DEE8', '#F6F5F3',
     '#EDC9CA', '#DA9092', '#C44E52', '#96393C'])

def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif'],
        'font.size': 9,
        'axes.titlesize': 9.5,
        'axes.labelsize': 9,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'axes.edgecolor': MUTED,
        'axes.labelcolor': INK,
        'axes.linewidth': 0.6,
        'text.color': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
    })

def despine(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)

TEXTWIDTH = 5.9   # inches, a4paper with 3 cm margins
