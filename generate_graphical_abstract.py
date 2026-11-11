"""
Generate the ECM graphical abstract: a single horizontal 3-panel composition.
Left  : skeletal engine + ORC flow schematic
Centre: advanced-exergy 2x2 bin matrix (AVOIDABLE/UNAVOIDABLE x ENDO/EXO)
Right : AV-EN ranking bar with the evaporator zero-avoidable callout

Output sized to Elsevier spec: 1328 x 531 px (W x H).
Writes graphical_abstract.png (raster) and .pdf (vector) into 01_ECM/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
import numpy as np

# ---- palette -------------------------------------------------------------
TERRA   = "#C0612F"   # exhaust
TEAL    = "#2E7D74"   # ORC working fluid
STEEL   = "#3A6EA5"   # seawater
INK     = "#22272B"
AVCOL   = "#C0612F"   # avoidable (warm = actionable)
UNCOL   = "#7C8A93"   # unavoidable (grey = structural)
PANEL   = "#F4F1EC"

DPI = 200
FIG_W, FIG_H = 1328 / DPI, 531 / DPI   # = 6.64 x 2.655 in

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor("white")

# three panels across the top ~80%, caption strip along the bottom ~20%
gs = fig.add_gridspec(1, 3, left=0.015, right=0.965, top=0.90, bottom=0.30,
                      wspace=0.20, width_ratios=[1.05, 1.0, 1.05])

# =========================================================================
# LEFT PANEL — flow schematic
# =========================================================================
axL = fig.add_subplot(gs[0, 0]); axL.set_xlim(0, 10); axL.set_ylim(0, 10)
axL.axis("off")
axL.set_title("System", fontsize=8.5, weight="bold", color=INK, pad=2)

def box(ax, x, y, w, h, label, fc, fs=6.3, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                linewidth=0, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", zorder=5)

# engine + turbocharger
box(axL, 0.4, 5.6, 3.4, 2.6, "MAN B&W\nS65ME-C8", INK)
box(axL, 4.3, 6.5, 2.4, 1.7, "Turbo-\ncharger", "#4A535A", fs=5.8)
# exhaust arrow (terracotta) to boiler
axL.add_patch(FancyArrow(6.8, 7.3, 1.7, 0.0, width=0.42, head_width=0.95,
                         head_length=0.55, length_includes_head=True, color=TERRA))
axL.text(7.0, 8.25, "245 °C", fontsize=5.3, color=TERRA, weight="bold")
box(axL, 6.6, 3.8, 2.9, 2.4, "Exhaust\nboiler /\nevaporator", TEAL, fs=5.8)
axL.text(7.4, 3.25, "150 °C out", fontsize=5.0, color=TERRA)
# ORC loop: turbine, condenser, pump
box(axL, 3.5, 2.0, 2.3, 1.7, "Turbine", TEAL, fs=6.0)
box(axL, 0.4, 2.0, 2.5, 1.7, "Condenser", STEEL, fs=6.0)
box(axL, 0.4, 0.2, 2.5, 1.2, "Pump", TEAL, fs=6.0)
# loop arrows
axL.add_patch(FancyArrow(6.6, 4.7, -0.7, 0.0, width=0.10, head_width=0.40,
                         head_length=0.35, length_includes_head=True, color=TEAL))
axL.annotate("", xy=(5.8, 2.85), xytext=(6.6, 4.4),
             arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4))
axL.add_patch(FancyArrow(3.5, 2.85, -0.55, 0.0, width=0.10, head_width=0.40,
                         head_length=0.35, length_includes_head=True, color=TEAL))
axL.annotate("", xy=(1.65, 2.0), xytext=(1.65, 1.4),
             arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.4))
axL.annotate("", xy=(7.7, 3.8), xytext=(1.65, 1.4),
             arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.0, alpha=0.5))
# seawater label
axL.text(1.65, 0.0, "29 °C  Bay of Bengal", fontsize=5.0, color=STEEL,
         weight="bold", ha="center")

# =========================================================================
# CENTRE PANEL — 2x2 advanced-exergy matrix
# =========================================================================
axC = fig.add_subplot(gs[0, 1]); axC.set_xlim(0, 10); axC.set_ylim(0, 10)
axC.axis("off")
axC.set_title("Advanced exergy 2×2", fontsize=8.5, weight="bold", color=INK, pad=2)

# quadrant frame
for (x0, x1) in [(1.2, 5.4), (5.6, 9.8)]:
    for (y0, y1) in [(1.2, 5.2), (5.4, 9.0)]:
        fc = PANEL
        axC.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                                    edgecolor="#C9C2B6", linewidth=0.8))
# axis labels
axC.text(3.3, 9.35, "AVOIDABLE", ha="center", fontsize=6.6, weight="bold", color=AVCOL)
axC.text(7.7, 9.35, "UNAVOIDABLE", ha="center", fontsize=6.6, weight="bold", color=UNCOL)
axC.text(0.65, 7.2, "ENDO-\nGENOUS", ha="center", va="center", fontsize=6.0,
         weight="bold", color=INK, rotation=90)
axC.text(0.65, 3.2, "EXO-\nGENOUS", ha="center", va="center", fontsize=6.0,
         weight="bold", color=INK, rotation=90)

# AV-EN bubbles (top-left quadrant) — area ~ kW
aven = [("Cyl", 1227.8), ("TC", 230.6), ("Turb", 215.5), ("Boil", 171.4)]
# layout positions inside the AV-EN cell
pos = [(2.6, 7.4), (4.3, 7.9), (4.5, 6.2), (3.1, 5.9)]
scale = 0.0019
for (lab, kw), (cx, cy) in zip(aven, pos):
    r = np.sqrt(kw * scale)
    axC.add_patch(Circle((cx, cy), r, facecolor=AVCOL, alpha=0.85, linewidth=0))
    axC.text(cx, cy, f"{lab}\n{kw:.0f}", ha="center", va="center",
             fontsize=4.8, color="white", weight="bold")
# UN-EN evaporator (top-right quadrant)
axC.add_patch(Circle((7.7, 7.0), np.sqrt(1261 * scale), facecolor=UNCOL,
                     alpha=0.9, linewidth=0))
axC.text(7.7, 7.0, "Evap\n1261", ha="center", va="center", fontsize=5.0,
         color="white", weight="bold")
axC.text(3.3, 1.7, "kW destruction sized by area", ha="center", fontsize=4.8,
         color="#6B6357", style="italic")

# =========================================================================
# RIGHT PANEL — AV-EN ranking bar
# =========================================================================
axR = fig.add_subplot(gs[0, 2])
axR.set_title("Improvement potential (AV-EN)", fontsize=7.6, weight="bold",
              color=INK, pad=2)
comps = ["Engine\ncylinders", "Turbo-\ncharger", "ORC\nturbine",
         "Exhaust\nboiler", "ORC\ncondenser", "ORC\nevaporator"]
vals = [1227.8, 230.6, 215.5, 171.4, 154.0, 0.0]
colors = [AVCOL] * 5 + ["#B9322E"]
ypos = np.arange(len(comps))[::-1]
axR.barh(ypos, vals, color=colors, height=0.66)
axR.set_yticks(ypos); axR.set_yticklabels(comps, fontsize=5.6)
axR.set_xlim(0, 1400)
axR.tick_params(axis="x", labelsize=5.5)
axR.set_xlabel("avoidable-endogenous  [kW]", fontsize=6.0)
for sp in ["top", "right"]:
    axR.spines[sp].set_visible(False)
for y, v in zip(ypos, vals):
    if v > 0:
        axR.text(v + 25, y, f"{v:.0f}", va="center", fontsize=5.2, color=INK)
# zero-avoidable callout on the evaporator
axR.annotate("zero avoidable\nat 2 K pinch", xy=(8, ypos[-1]), xytext=(430, ypos[-1] - 0.05),
             fontsize=5.4, color="#B9322E", weight="bold", va="center",
             arrowprops=dict(arrowstyle="-|>", color="#B9322E", lw=1.0))

# =========================================================================
# CAPTION STRIP
# =========================================================================
fig.text(0.5, 0.155,
         "Advanced exergy decomposition locates improvable destruction on the engine side; "
         "the ORC evaporator is fully unavoidable at design pinch.",
         ha="center", va="center", fontsize=6.6, color=INK, wrap=False)
fig.text(0.5, 0.055,
         "MAN B&W S65ME-C8 + ORC  ·  85% MCR  ·  29 °C Indian Ocean seawater",
         ha="center", va="center", fontsize=5.8, color="#6B6357", style="italic")

out_dir = os.path.join(os.path.dirname(__file__), "..",
                       "journal_submissions", "manuscripts", "01_ECM")
out_dir = os.path.abspath(out_dir)
png = os.path.join(out_dir, "graphical_abstract.png")
pdf = os.path.join(out_dir, "graphical_abstract.pdf")
fig.savefig(png, dpi=DPI, facecolor="white")
fig.savefig(pdf, facecolor="white")
print("wrote", png)
print("wrote", pdf)
