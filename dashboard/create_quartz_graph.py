import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

# ── journal style ─────────────────────────────────────────────────────────────
rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Times New Roman', 'Georgia', 'serif'],
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       150,
})

# ── palette ───────────────────────────────────────────────────────────────────
BG      = '#FFFFFF'
TEXT    = '#1a1a2e'
GREY_LG = '#e8e8e8'   # grid lines
GREY_MG = '#aaaaaa'   # secondary text / spines
GREEN   = '#2ecc71'
RED     = '#e74c3c'
BLUE    = '#3498db'
ORANGE  = '#e67e22'

# ── data ─────────────────────────────────────────────────────────────────────
r2_p1  = [0.9842,0.9837,0.9826,0.9836,0.9845,0.9844,0.9846,0.9842,0.9852,0.9834,0.9843,0.9857]
r2_p2  = [0.9772,0.9763,0.9755,0.9765,0.9810,0.9810,0.9796,0.9791,0.9790,0.9826,0.9838,0.9839]
r2_p3  = [0.9750,0.9756,0.9767,0.9742,0.9780,0.9791,0.9802,0.9798,0.9827,0.9822,0.9826,0.9830]
r2_all = r2_p1 + r2_p2 + r2_p3

mae_p1  = [17.42,17.68,18.36,17.50,16.89,17.13,16.94,17.11,16.13,17.07,16.57,16.22]
mae_p2  = [20.44,20.71,21.11,20.74,18.34,18.46,18.94,19.43,19.60,18.02,17.39,17.46]
mae_p3  = [21.80,21.68,21.09,21.98,19.55,19.21,18.63,18.76,17.60,18.16,17.88,17.69]
mae_all = mae_p1 + mae_p2 + mae_p3

temp_p1 = [36.09,39.10,41.16,41.39,40.71,40.99,35.80,39.12,39.90,41.22,40.91,39.35]
temp_p2 = [36.60,38.48,40.38,40.66,40.40,38.64,36.42,38.46,39.15,39.51,38.83,37.95]
temp_p3 = [36.11,41.39,38.35,41.28,40.47,40.58,37.37,39.59,40.82,40.52,38.31,36.99]

vmin_p1 = [2.976,2.978,2.892,3.053,3.038,3.020,3.012,2.998,3.021,3.013,3.001,3.016]
vmin_p2 = [3.004,3.009,2.991,2.942,3.021,3.084,3.030,2.898,2.993,2.964,2.989,3.062]
vmin_p3 = [2.996,2.996,3.046,3.045,2.979,3.043,3.027,3.017,3.021,2.877,2.963,3.047]
vmin_all = vmin_p1 + vmin_p2 + vmin_p3

cell_labels = [f'P{p}S{s}' for p in range(1,4) for s in range(1,13)]
x = np.arange(36)

WEAKEST   = 29  # P3S10
STRONGEST = 17  # P2S6

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30,
                      left=0.07, right=0.97, top=0.88, bottom=0.07)

ax_r2   = fig.add_subplot(gs[0, 0])
ax_mae  = fig.add_subplot(gs[0, 1])
ax_heat = fig.add_subplot(gs[1, 0])
ax_vmin = fig.add_subplot(gs[1, 1])

def style_ax(ax, title, ylabel='', xlabel=''):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8, length=3)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(GREY_MG)
        ax.spines[spine].set_linewidth(0.8)
    ax.set_title(title, fontsize=11.5, fontweight='bold', color=TEXT, pad=10,
                 loc='left')
    if ylabel: ax.set_ylabel(ylabel, fontsize=9.5, color=TEXT)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9.5, color=TEXT)
    ax.yaxis.grid(True, color=GREY_LG, linewidth=0.6, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

def group_dividers(ax, ypos, label_y, ylim_top):
    for xv in [11.5, 23.5]:
        ax.axvline(xv, color=GREY_MG, linewidth=0.7, linestyle=':', zorder=2)
    for gx, lbl in [(5.5, 'Pack 1'), (17.5, 'Pack 2'), (29.5, 'Pack 3')]:
        ax.text(gx, label_y, lbl, color=GREY_MG, fontsize=8,
                ha='center', va='bottom', style='italic')

# ── PLOT 1 — R² per cell ─────────────────────────────────────────────────────
colors_r2 = [GREEN if v > 0.98 else BLUE if v > 0.95 else ORANGE for v in r2_all]
ax_r2.bar(x, r2_all, color=colors_r2, width=0.72, zorder=3,
          edgecolor='white', linewidth=0.3)
ax_r2.axhline(0.98, color=RED, linestyle='--', linewidth=1.2, zorder=4,
              label='R² = 0.98 target')
style_ax(ax_r2, '(a)  R² per Cell — Quartz WLTP (36 Real Cells)', ylabel='R²')
ax_r2.set_ylim(0.969, 0.990)
ax_r2.set_xlim(-1, 36)
group_dividers(ax_r2, None, 0.9893, 0.990)
ax_r2.set_xticks(x[::3])
ax_r2.set_xticklabels(cell_labels[::3], rotation=45, ha='right', fontsize=7)
ax_r2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))

legend_r2 = [
    mpatches.Patch(color=GREEN,  label='R² > 0.980'),
    mpatches.Patch(color=BLUE,   label='R² > 0.950'),
    plt.Line2D([0],[0], color=RED, linestyle='--', linewidth=1.2, label='Target R²=0.98'),
]
ax_r2.legend(handles=legend_r2, fontsize=8, loc='lower right',
             framealpha=0.9, edgecolor=GREY_MG, facecolor=BG)

# annotate best and worst
for idx in [r2_all.index(max(r2_all)), r2_all.index(min(r2_all))]:
    offset = 0.00012
    ax_r2.text(idx, r2_all[idx] + offset, f'{r2_all[idx]:.4f}',
               ha='center', va='bottom', fontsize=6.5,
               color=TEXT, fontweight='bold')

# ── PLOT 2 — MAE per cell ────────────────────────────────────────────────────
colors_mae = [GREEN if v < 20 else RED for v in mae_all]
ax_mae.bar(x, mae_all, color=colors_mae, width=0.72, zorder=3,
           edgecolor='white', linewidth=0.3)
ax_mae.axhline(20, color=RED, linestyle='--', linewidth=1.2, zorder=4)
style_ax(ax_mae, '(b)  MAE per Cell — Target < 20 mV', ylabel='MAE (mV)')
ax_mae.set_ylim(13.5, 24.5)
ax_mae.set_xlim(-1, 36)
group_dividers(ax_mae, None, 24.0, 24.5)
ax_mae.set_xticks(x[::3])
ax_mae.set_xticklabels(cell_labels[::3], rotation=45, ha='right', fontsize=7)

ax_mae.text(35.5, 20.4, '20 mV\nstandard', ha='right', va='bottom',
            fontsize=7.5, color=RED, style='italic')

legend_mae = [
    mpatches.Patch(color=GREEN, label='MAE < 20 mV'),
    mpatches.Patch(color=RED,   label='MAE >= 20 mV  (fail)'),
]
ax_mae.legend(handles=legend_mae, fontsize=8, loc='upper right',
              framealpha=0.9, edgecolor=GREY_MG, facecolor=BG)

worst_idx = mae_all.index(max(mae_all))
ax_mae.annotate(f'{mae_all[worst_idx]:.1f} mV',
                xy=(worst_idx, mae_all[worst_idx]),
                xytext=(worst_idx + 3, mae_all[worst_idx] + 0.8),
                color=TEXT, fontsize=7.5, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=GREY_MG, lw=0.9))

# ── PLOT 3 — Temperature heatmap ─────────────────────────────────────────────
temp_grid = np.array([temp_p1, temp_p2, temp_p3])

cmap_thermal = LinearSegmentedColormap.from_list(
    'thermal_journal',
    ['#d6eaf8', '#85c1e9', '#27ae60', '#f39c12', '#c0392b'])

im = ax_heat.imshow(temp_grid, cmap=cmap_thermal, aspect='auto',
                    vmin=34, vmax=43, interpolation='nearest')

ax_heat.set_facecolor(BG)
ax_heat.set_title('(c)  Cell Temperature Distribution (°C)',
                  fontsize=11.5, fontweight='bold', color=TEXT, pad=10, loc='left')
ax_heat.set_yticks([0, 1, 2])
ax_heat.set_yticklabels(['Pack 1', 'Pack 2', 'Pack 3'],
                         color=TEXT, fontsize=9)
ax_heat.set_xticks(range(12))
ax_heat.set_xticklabels([f'S{i+1}' for i in range(12)], color=TEXT, fontsize=8)
for spine in ax_heat.spines.values():
    spine.set_edgecolor(GREY_MG)
    spine.set_linewidth(0.8)

# cell value text
for r in range(3):
    for c in range(12):
        v = temp_grid[r, c]
        txt_col = 'white' if v > 40 else TEXT
        ax_heat.text(c, r, f'{v:.1f}', ha='center', va='center',
                     fontsize=7, color=txt_col, fontweight='bold')

# special cell borders
specials = {
    (2, 9): ('P3S10\nWeakest', RED),
    (2, 1): ('P3S2\nHottest', '#c0392b'),
    (1, 5): ('P2S6\nStrongest', '#27ae60'),
}
for (rr, cc), (lbl, col) in specials.items():
    ax_heat.add_patch(mpatches.FancyBboxPatch(
        (cc - 0.48, rr - 0.48), 0.96, 0.96,
        boxstyle='round,pad=0.05', linewidth=2.0,
        edgecolor=col, facecolor='none', zorder=5))
    ax_heat.text(cc, rr - 0.38, lbl, ha='center', va='top',
                 fontsize=5.5, color=col, fontweight='bold')

cb = plt.colorbar(im, ax=ax_heat, pad=0.015, fraction=0.032,
                  orientation='vertical')
cb.ax.tick_params(labelsize=8, colors=TEXT)
cb.set_label('Temperature (°C)', fontsize=9, color=TEXT)
cb.outline.set_edgecolor(GREY_MG)

# ── PLOT 4 — V_min per cell ───────────────────────────────────────────────────
colors_v = []
for i, v in enumerate(vmin_all):
    if i == WEAKEST:     colors_v.append(RED)
    elif i == STRONGEST: colors_v.append(GREEN)
    else:                colors_v.append(BLUE)

ax_vmin.bar(x, vmin_all, color=colors_v, width=0.72, zorder=3,
            edgecolor='white', linewidth=0.3)
ax_vmin.axhline(3.0, color=RED, linestyle='--', linewidth=1.2, zorder=4)
style_ax(ax_vmin, '(d)  Minimum Cell Voltage — Weakest Cell Detection',
         ylabel='V$_{min}$ (V)')
ax_vmin.set_ylim(2.81, 3.13)
ax_vmin.set_xlim(-1, 36)
group_dividers(ax_vmin, None, 3.12, 3.13)
ax_vmin.set_xticks(x[::3])
ax_vmin.set_xticklabels(cell_labels[::3], rotation=45, ha='right', fontsize=7)
ax_vmin.text(35.5, 3.005, '3.0 V threshold', ha='right', va='bottom',
             fontsize=7.5, color=RED, style='italic')

for idx, lbl, xoff, yoff in [
    (WEAKEST,   'P3S10\n2.877 V\n(weakest)',   +4.0, -0.015),
    (STRONGEST, 'P2S6\n3.084 V\n(strongest)',  -4.0, +0.010),
]:
    ax_vmin.annotate(lbl,
                     xy=(idx, vmin_all[idx]),
                     xytext=(idx + xoff, vmin_all[idx] + yoff),
                     color=TEXT, fontsize=7.5, fontweight='bold', ha='center',
                     arrowprops=dict(arrowstyle='->', color=GREY_MG, lw=0.9))

legend_v = [
    mpatches.Patch(color=RED,   label='Weakest cell (P3S10)'),
    mpatches.Patch(color=GREEN, label='Strongest cell (P2S6)'),
    mpatches.Patch(color=BLUE,  label='Other cells'),
]
ax_vmin.legend(handles=legend_v, fontsize=8, loc='lower right',
               framealpha=0.9, edgecolor=GREY_MG, facecolor=BG)

# ── supra-title ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.955,
         'OpenCATHODE Stack — Quartz WLTP Real Validation',
         ha='center', color=TEXT, fontsize=17, fontweight='bold',
         fontfamily='serif')
fig.text(0.5, 0.928,
         'R² = 0.9810  |  MAE = 18.6 mV  |  36/36 Cells  |  634,450 Real Datapoints',
         ha='center', color=GREY_MG, fontsize=11, fontstyle='italic',
         fontfamily='serif')

# thin rule under supra-title
fig.add_artist(plt.Line2D([0.07, 0.97], [0.916, 0.916],
                           transform=fig.transFigure,
                           color=GREY_MG, linewidth=0.8))

# ── save ─────────────────────────────────────────────────────────────────────
out = 'dashboard/quartz_validation_graph.png'
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()
print(f'Saved → {out}')
