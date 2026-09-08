"""Schematic of the bleeding x ischaemic trade-off plane (Chapter 3, "The trade-off plane").

Not data: a diagram of what the four quadrants mean, using the same axis convention as
CATE_tradeoff_plane.png -- both axes are the CATE of ABBREVIATING, so >0 always means
"abbreviating helps on this endpoint" and the ischaemic axis is the weighted composite
(0.4 MI, 0.4 stroke, 0.2 cardiovascular death).

Run:  .venv/bin/python markdown_docs/thesis/images/make_tradeoff_zones.py
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = __file__.replace('make_tradeoff_zones.py', 'CATE_tradeoff_zones.png')

INK      = '#0b0b0b'   # primary text
INK_2    = '#52514e'   # secondary text
MUTED    = '#898781'   # axis / hints
SURFACE  = '#fcfcfb'

# status palette: the zone tint carries the flavour, the label carries the meaning
GOOD, WARNING, SERIOUS, CRITICAL = '#0ca30c', '#fab219', '#ec835a', '#d03b3b'
TINT = {GOOD: '#e8f5e8', WARNING: '#fdf3dc', SERIOUS: '#fbeade', CRITICAL: '#f8e4e4'}

L = 1.0   # half-width of the schematic plane (arbitrary units: the diagram has no scale)

fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# ---------------------------------------------------------------- the four zones
zones = [
    # (x0, y0, colour, title, body, verdict)
    (0, 0, GOOD, 'WIN - WIN',
     'abbreviating prevents bleeds\nAND ischaemic events',
     'abbreviate: nothing to trade'),
    (-L, 0, SERIOUS, 'REVERSE TRADE-OFF',
     'abbreviating prevents ischaemic\nevents BUT causes bleeds',
     'the trade-off, mirrored'),
    (0, -L, WARNING, 'TRADE-OFF',
     'abbreviating prevents bleeds\nBUT costs ischaemic events',
     'depends on the patient'),
    (-L, -L, CRITICAL, 'LOSE - LOSE',
     'abbreviating worsens bleeding\nAND ischaemia',
     'keep the prolonged therapy'),
]

for x0, y0, colour, title, body, verdict in zones:
    ax.add_patch(Rectangle((x0, y0), L, L, facecolor=TINT[colour], edgecolor='none', zorder=0))
    if colour is WARNING:                    # the only quadrant a personalised rule would need
        ax.add_patch(Rectangle((x0 + .012, y0 + .012), L - .024, L - .024, facecolor='none',
                               edgecolor=colour, lw=2.0, zorder=1))
    cx, cy = x0 + L / 2, y0 + L / 2
    ax.text(cx, cy + .26, title, ha='center', va='center', color=INK,
            fontsize=13, fontweight='bold', zorder=3)
    ax.text(cx, cy + .04, body, ha='center', va='center', color=INK_2,
            fontsize=10.5, linespacing=1.5, zorder=3)
    ax.text(cx, cy - .27, verdict, ha='center', va='center', color=colour,
            fontsize=10, fontstyle='italic', fontweight='bold', zorder=3)

# ---------------------------------------------------------------- the win-win diagonal
ax.annotate('', xy=(.95, .95), xytext=(.74, .74), zorder=2,
            arrowprops=dict(arrowstyle='-|>', color=GOOD, lw=1.4, ls=(0, (5, 3)),
                            shrinkA=0, shrinkB=0))
ax.text(.72, .70, 'win-win direction (+45°)', color=GOOD, fontsize=8.5,
        ha='center', va='top', zorder=3)

# ---------------------------------------------------------------- the axes through zero
for xy, xytext in [((L, 0), (-L, 0)), ((0, L), (0, -L))]:
    ax.annotate('', xy=xy, xytext=xytext, zorder=2,
                arrowprops=dict(arrowstyle='<|-|>', color=INK, lw=1.3,
                                shrinkA=0, shrinkB=0))
ax.plot(0, 0, 'o', ms=5, color=INK, zorder=4)
ax.text(.035, -.075, '0 = no effect', color=MUTED, fontsize=9, ha='left', va='top', zorder=4)

for x, sign in [(L - .05, '+'), (-L + .05, '−')]:
    ax.text(x, .045, sign, color=INK, fontsize=13, ha='center', va='bottom', zorder=4)
for y, sign in [(L - .05, '+'), (-L + .05, '−')]:
    ax.text(.03, y, sign, color=INK, fontsize=13, ha='left', va='center', zorder=4)

ax.set_xlabel('CATE on bleeding  —  benefit of abbreviating   ( > 0 : fewer bleeds )',
              fontsize=11, color=INK, labelpad=12)
ax.set_ylabel('weighted ischaemic CATE  —  benefit of abbreviating   ( > 0 : fewer events )',
              fontsize=11, color=INK, labelpad=12)

ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
for s in ax.spines.values():
    s.set_visible(False)

ax.set_title('The trade-off plane: what each quadrant means',
             fontsize=14, color=INK, pad=16)

fig.text(.5, .028,
         'Both axes are the effect of ABBREVIATING the therapy, so positive always means "abbreviating helps".\n'
         'The ischaemic axis is the weighted composite (0.4 MI, 0.4 stroke, 0.2 cardiovascular death).\n'
         'Only in the bottom-right quadrant does the answer genuinely depend on the patient - a single trial-wide\n'
         'rule is enough everywhere else, which is why that quadrant is the one a personalised policy has to find.',
         ha='center', va='bottom', fontsize=9.2, color=INK_2, linespacing=1.6)

fig.subplots_adjust(left=.075, right=.975, top=.925, bottom=.20)
fig.savefig(OUT, facecolor=SURFACE)
print('wrote', OUT)