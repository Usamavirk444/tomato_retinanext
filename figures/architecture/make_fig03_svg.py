"""
make_fig03_svg.py
─────────────────
Figure 3 — RetinaNeXt in detail.

Layout follows the reference composition: one wide forward-path panel on top,
two detail panels below, compact proportions, isometric 3D feature blocks.

Fixed relative to the reference image:
    "Six featuure scales"                     typo
    duplicated "Input 640 x 640" labels       drawn once
    "192 ch, stride 8384 ch, stride 16..."    run-together stage sub-labels
    reversed panel-B legend                   grey = top-down, red = bottom-up
    duplicated panel-C caption                removed

Every number is read from the code, not recalled:
    input 640x640                     configs/experiment.yaml:45
    stem 96 ch /4, unused by neck     convnext_backbone.py
    C3/C4/C5 = 192/384/768            convnext_backbone.py
    BiFPN 3 repeats, 256 ch, C3-C5    bifpn.py
    DW-sep conv + BN + SiLU           bifpn.py
    fast norm  w = relu(w) + 1e-4     bifpn.py
    LastLevelP6P7 after the BiFPN     retinanet_hybrid.py
    CBAMNeck(num_levels=5, ch=256)    cbam.py
    channel reduction 16, spatial 7x7 cbam.py
    9 anchors per location            retinanet_hybrid.py
    35.20 M parameters, 26.1 FPS      outputs/results/fps_benchmark.csv

Usage:
    python make_fig03_svg.py       -> Fig03_RetinaNeXt_architecture.svg
"""

import math
import os

# ── palette ──────────────────────────────────────────────────────────
NAVY   = "#1F2A44"; NAVY_T  = "#36425F"; NAVY_S  = "#121829"
GOLD   = "#DFB13C"; GOLD_T  = "#EFC866"; GOLD_S  = "#A07C16"
BLUE   = "#C5D9F1"; BLUE_T  = "#DEEAF7"; BLUE_S  = "#7FA6CF"
CREAM  = "#F6EDC8"; CREAM_T = "#FBF6E2"; CREAM_S = "#C9A94A"
GREY   = "#E8E8E8"; GREY_T  = "#F5F5F5"; GREY_S  = "#A8A8A8"
WHITE  = "#FFFFFF"; WHITE_T = "#FFFFFF"; WHITE_S = "#9A9A9A"
BG     = "#F7F8FA"; PANEL   = "#FFFFFF"; EDGE = "#4472C4"
ACCENT = "#C0504D"; TEXT    = "#111111"; MUTED = "#555555"

DEPTH = 13
out = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=11, weight="normal", fill=TEXT, anchor="start"):
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
               f'font-family="Helvetica,Arial,sans-serif" font-size="{size}" '
               f'font-weight="{weight}" fill="{fill}">{esc(s)}</text>')


def cuboid(x, y, w, h, face, top, side, lw=0.9):
    d = DEPTH
    out.append(f'<polygon points="{x:.1f},{y:.1f} {x+d:.1f},{y-d:.1f} '
               f'{x+w+d:.1f},{y-d:.1f} {x+w:.1f},{y:.1f}" '
               f'fill="{top}" stroke="{side}" stroke-width="{lw}"/>')
    out.append(f'<polygon points="{x+w:.1f},{y:.1f} {x+w+d:.1f},{y-d:.1f} '
               f'{x+w+d:.1f},{y-d+h:.1f} {x+w:.1f},{y+h:.1f}" '
               f'fill="{side}" stroke="{side}" stroke-width="{lw}"/>')
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
               f'fill="{face}" stroke="{side}" stroke-width="{lw}"/>')


def slab(spatial, channels, s=0.52, c=1.60):
    return max(14, math.sqrt(channels) * 1.05 * c), max(24, math.sqrt(spatial) * 11.0 * s)


def group(x, ym, spatial, channels, n, face, top, side, gap=8, s=0.52, c=1.60):
    w, h = slab(spatial, channels, s, c)
    for i in range(n):
        cuboid(x + i * (w + gap), ym - h / 2, w, h, face, top, side)
    return x + n * w + (n - 1) * gap, h


def arrow(x1, y, x2, colour=ACCENT, marker="ahr", width=2.0):
    out.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2-7:.1f}" y2="{y:.1f}" '
               f'stroke="{colour}" stroke-width="{width}" marker-end="url(#{marker})"/>')


def panel(x, y, w, h, stroke=EDGE, sw=1.5):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
               f'fill="{PANEL}" stroke="{stroke}" stroke-width="{sw}"/>')


def rbox(x, y, w, h, fill, stroke, lines, size=11, weight="bold"):
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
               f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="0.9"/>')
    n = len(lines)
    first = y + h / 2 - (n - 1) * 7 + 4
    for i, ln in enumerate(lines):
        text(x + w / 2, first + i * 14, ln, size, weight, TEXT, "middle")


# ════════════════════════════════════════════════════════════════════
W, H = 1410, 792
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}">')
out.append('<defs>'
           f'<marker id="ahr" markerWidth="7" markerHeight="7" refX="6" refY="3" '
           f'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="{ACCENT}"/></marker>'
           f'<marker id="ahg" markerWidth="7" markerHeight="7" refX="6" refY="3" '
           f'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#808080"/></marker>'
           '</defs>')
out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

text(W / 2, 32, "RetinaNeXt: ConvNeXt-Tiny backbone, BiFPN neck and CBAM "
                "attention on a RetinaNet head", 22, "bold", TEXT, "middle")
text(W / 2, 54, "Block height encodes spatial resolution; block width encodes "
                "channel count. Spatial sizes are for a 640 x 640 input.",
     11, "normal", MUTED, "middle")

# ── panel A: forward path ────────────────────────────────────────────
PX, PY, PW, PH = 18, 78, W - 36, 352
panel(PX, PY, PW, PH, ACCENT, 2.0)
text(PX + 22, PY + 30, "A. Forward path", 17, "bold", TEXT)
text(PX + 22, PY + 50, "35.20 M parameters, 26.1 FPS, 9 foreground classes, "
     "9 anchors per location", 12, "normal", MUTED)

ym = 254
LBL, S1, S2, S3 = 338, 354, 369, 384          # label baselines, all aligned

# input
cuboid(42, ym - 28, 56, 56, WHITE, WHITE_T, WHITE_S)
text(70, LBL, "Input", 11.5, "bold", TEXT, "middle")
text(70, S1, "640 x 640", 10, "normal", MUTED, "middle")
arrow(116, ym, 152)

# backbone: stem + three tapped stages
stages = [(160, 96, "Stem + stage 1", "96 ch, stride 4", True),
          (80, 192, "Stage 2 - C3", "192 ch, stride 8", False),
          (40, 384, "Stage 3 - C4", "384 ch, stride 16", False),
          (20, 768, "Stage 4 - C5", "768 ch, stride 32", False)]
bx, cx_list, geom = 152, [], []
for k, (sp, ch, nm, sub, unused) in enumerate(stages):
    w, h = slab(sp, ch)
    face, top, side = (GREY, GREY_T, GREY_S) if unused else (NAVY, NAVY_T, NAVY_S)
    cuboid(bx, ym - h / 2, w, h, face, top, side)
    cx = bx + w / 2
    cx_list.append(cx)
    geom.append((cx, bx + w + DEPTH, ym + h / 2))
    text(cx, LBL, nm, 11.5, "bold", TEXT, "middle")
    text(cx, S1, sub, 10, "normal", MUTED, "middle")
    if k < len(stages) - 1:                 # sequential arrow into the next stage
        arrow(bx + w + DEPTH + 5, ym, bx + 100)
    bx += 100
text(cx_list[0], S2, "not used by the neck", 10, "normal", ACCENT, "middle")
bmid = (cx_list[0] + cx_list[-1]) / 2
text(bmid, 168, "ConvNeXt-Tiny", 13.5, "bold", TEXT, "middle")
text(bmid, 183, "7x7 depthwise convs, depths 3 / 3 / 9 / 3", 10.5, "normal",
     MUTED, "middle")

back_end = bx - 100 + slab(20, 768)[0] + DEPTH
arrow(back_end + 8, ym, back_end + 76)

# BiFPN
bf = back_end + 76
bf_end, bf_h = group(bf, ym, 250, 256, 3, GOLD, GOLD_T, GOLD_S, gap=9, c=1.45)
bf_mid = (bf + bf_end) / 2
text(bf_mid, 183, "BiFPN x 3", 13.5, "bold", TEXT, "middle")
text(bf_mid, LBL, "C3-C5 only, 256 ch", 10, "normal", MUTED, "middle")
text(bf_mid, S1, "depthwise-separable conv", 10, "normal", MUTED, "middle")
text(bf_mid, S2, "+ BN + SiLU,", 10, "normal", MUTED, "middle")
text(bf_mid, S3, "learnable fusion weights", 10, "normal", MUTED, "middle")

# C3 and C4 tap into the BiFPN alongside C5: all three levels are fused
for (cx_s, _, bot), run_y, riser_x, entry_y in (
        (geom[1], 318, bf - 54, ym + 34),      # Stage 2 -> C3
        (geom[2], 302, bf - 28, ym + 13)):     # Stage 3 -> C4
    out.append(f'<path d="M {cx_s:.1f} {bot:.1f} L {cx_s:.1f} {run_y} '
               f'L {riser_x:.1f} {run_y} L {riser_x:.1f} {entry_y:.1f} '
               f'L {bf - 3:.1f} {entry_y:.1f}" fill="none" stroke="{ACCENT}" '
               f'stroke-width="1.6" marker-end="url(#ahr)"/>')
arrow(bf_end + DEPTH + 6, ym, bf_end + DEPTH + 44)

# LastLevelP6P7
p6 = bf_end + DEPTH + 44
p6_end, _ = group(p6, ym, 100, 256, 2, GOLD, GOLD_T, GOLD_S, gap=9, c=1.45)
p6_mid = (p6 + p6_end) / 2
text(p6_mid, 168, "LastLevelP6P7", 13.5, "bold", TEXT, "middle")
text(p6_mid, 183, "strided conv", 10.5, "normal", ACCENT, "middle")
text(p6_mid, 197, "P6 10x10, P7 5x5", 10.5, "normal", MUTED, "middle")
text(p6_mid, LBL, "P6 10x10, P7 5x5", 10, "normal", MUTED, "middle")
text(p6_mid, S1, "strided conv from P5,", 10, "normal", ACCENT, "middle")
text(p6_mid, S2, "after the BiFPN", 10, "normal", ACCENT, "middle")
arrow(p6_end + DEPTH + 6, ym, p6_end + DEPTH + 44)

# CBAM
cb = p6_end + DEPTH + 44
cb_end, _ = group(cb, ym, 250, 256, 5, CREAM, CREAM_T, CREAM_S, gap=6, c=0.80)
cb_mid = (cb + cb_end) / 2
text(cb_mid, 183, "CBAM", 13.5, "bold", TEXT, "middle")
text(cb_mid, LBL, "all five levels", 10, "normal", MUTED, "middle")
text(cb_mid, S1, "P3-P7", 10, "normal", MUTED, "middle")
arrow(cb_end + DEPTH + 6, ym, cb_end + DEPTH + 42)

# head
hx = cb_end + DEPTH + 42
cuboid(hx, ym - 48, 92, 96, BLUE, BLUE_T, BLUE_S)
text(hx + 46, 168, "RetinaNet head", 13.5, "bold", TEXT, "middle")
text(hx + 46, 183, "unchanged from the baseline", 10.5, "normal", MUTED, "middle")
text(hx + 46, ym - 20, "Shared", 11.5, "bold", TEXT, "middle")
text(hx + 46, ym - 6, "subnets", 11.5, "bold", TEXT, "middle")
text(hx + 46, ym + 10, "4 x conv 3x3", 10, "normal", MUTED, "middle")
text(hx + 46, ym + 24, "(256) each", 10, "normal", MUTED, "middle")
arrow(hx + 92 + DEPTH + 6, ym, hx + 92 + DEPTH + 42)

ox = hx + 92 + DEPTH + 48
text(ox, ym - 14, "Classification", 11.5, "bold")
text(ox, ym + 1, "9 classes x 9 anchors", 10, "normal", MUTED)
text(ox, ym + 24, "Box regression", 11.5, "bold")
text(ox, ym + 39, "4 offsets x 9 anchors", 10, "normal", MUTED)

text(W / 2, PY + PH - 16, "Anchor sizes per level: 32/40/50, 64/80/101, "
     "128/161/203, 256/322/406, 512/645/812 at ratios 0.5, 1.0 and 2.0. "
     "Sigmoid focal loss, alpha = 0.25, gamma = 2.0.",
     10.5, "normal", MUTED, "middle")

# ── panel B: one BiFPN cell ──────────────────────────────────────────
QY, QH = 446, 330
panel(18, QY, 686, QH)
text(40, QY + 32, "B. One BiFPN cell", 17, "bold", TEXT)
text(40, QY + 53, "Repeated three times. Fusion spans P3-P5 only.", 12,
     "normal", MUTED)

c_in, c_td, c_out = 118, 330, 572
r5, r4, r3 = QY + 106, QY + 169, QY + 232
BW, BH = 96, 30
GY = "#808080"

for cx, cy, lab, fill, strk in [
        (c_in, r5, "P5 in", BLUE, BLUE_S), (c_in, r4, "P4 in", BLUE, BLUE_S),
        (c_in, r3, "P3 in", BLUE, BLUE_S),
        (c_td, r4, "P4 td", GOLD, GOLD_S),
        (c_out, r5, "P5 out", CREAM, CREAM_S),
        (c_out, r4, "P4 out", CREAM, CREAM_S),
        (c_out, r3, "P3 out", CREAM, CREAM_S)]:
    rbox(cx - BW / 2, cy - BH / 2, BW, BH, fill, strk, [lab], 11.5)


def edge(x1, y1, x2, y2, colour=GY, dash=None, marker="ahg"):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
               f'stroke="{colour}" stroke-width="1.6"{d} '
               f'marker-end="url(#{marker})"/>')


L, R = BW / 2, BW / 2
# top-down (grey)
edge(c_in + L, r5, c_td - R - 8, r4 - 6)                 # P5 in -> P4 td
edge(c_in + L, r4, c_td - R - 8, r4)                     # P4 in -> P4 td
edge(c_td, r4 + BH / 2, c_td, r3 - BH / 2 - 8)           # P4 td -> P3 out
edge(c_in + L, r3, c_out - R - 8, r3)                    # P3 in -> P3 out
edge(c_in + L, r5, c_out - R - 8, r5)                    # P5 in -> P5 out
# bottom-up (red): P3 out -> P4 out -> P5 out, same column
edge(c_out, r3 - BH / 2, c_out, r4 + BH / 2 + 8, ACCENT, None, "ahr")
edge(c_out, r4 - BH / 2, c_out, r5 + BH / 2 + 8, ACCENT, None, "ahr")
# skip edge (dashed grey): P4 in -> P4 out, routed clear of P4 td
out.append(f'<path d="M {c_in:.0f} {r4+BH/2:.0f} L {c_in:.0f} {r4+38:.0f} '
           f'L {c_out-R-28:.0f} {r4+38:.0f} L {c_out-R-28:.0f} {r4+9:.0f} '
           f'L {c_out-R-8:.0f} {r4+9:.0f}" '
           f'fill="none" stroke="{GY}" stroke-width="1.4" stroke-dasharray="5 4" '
           f'marker-end="url(#ahg)"/>')

LY = QY + 283
for i, (col, dash, lab) in enumerate([(GY, None, "Top-down paths"),
                                      (ACCENT, None, "Bottom-up paths"),
                                      (GY, "5 4", "Skip edge paths")]):
    lx = 48 + i * 210
    d = f' stroke-dasharray="{dash}"' if dash else ''
    mk = "ahr" if col == ACCENT else "ahg"
    out.append(f'<line x1="{lx}" y1="{LY}" x2="{lx+40}" y2="{LY}" stroke="{col}" '
               f'stroke-width="1.6"{d} marker-end="url(#{mk})"/>')
    text(lx + 50, LY + 4, lab, 11, "normal", MUTED)
text(48, QY + 312, "Each edge carries a learnable weight, normalised as "
     "w = relu(w) + 1e-4 divided by the sum.", 10.5, "normal", MUTED)

# ── panel C: CBAM ────────────────────────────────────────────────────
CXP = 720
panel(CXP, QY, W - CXP - 18, QH)
text(CXP + 22, QY + 32, "C. CBAM, applied at every level", 17, "bold", TEXT)
text(CXP + 22, QY + 53, "Channel attention first, then spatial attention. "
     "0.041 M parameters total.", 12, "normal", MUTED)

cy = QY + 162
cuboid(CXP + 32, cy - 45, 44, 90, CREAM, CREAM_T, CREAM_S)
text(CXP + 54 + DEPTH / 2, cy + 68, "256 ch", 10.5, "normal", MUTED, "middle")
arrow(CXP + 92, cy, CXP + 126, GY, "ahg", 1.8)

chain = [(CXP + 126, 82, ["Avg", "pool", "1x1"], BLUE, BLUE_S, None),
         (CXP + 224, 118, ["Shared MLP", "256-16-256"], BLUE, BLUE_S,
          ["sigmoid", "channel gate"]),
         (CXP + 358, 82, ["Mean", "and", "max"], BLUE, BLUE_S, None),
         (CXP + 456, 86, ["Conv", "7x7"], BLUE, BLUE_S, None),
         (CXP + 558, 94, ["sigmoid", "spatial"], BLUE, BLUE_S, ["spatial gate"])]
for bxp, bw, lines, fill, strk, below in chain:
    rbox(bxp, cy - 40, bw, 80, fill, strk, lines, 11.5)
    if below:
        for j, ln in enumerate(below):
            text(bxp + bw / 2, cy + 58 + j * 14, ln, 10.5, "normal", MUTED,
                 "middle")
for i in range(len(chain) - 1):
    x1 = chain[i][0] + chain[i][1]
    arrow(x1 + 4, cy, chain[i + 1][0] - 2, GY, "ahg", 1.8)

text(CXP + 22, QY + 265, "Both gates multiply the feature map element-wise. "
     "Channel reduction ratio is 16, so 256 - 16 - 256.", 10.5, "normal", MUTED)
text(CXP + 22, QY + 285, "Applied after the neck, to all five levels P3-P7. "
     "The backbone is never touched by CBAM.", 10.5, "normal", MUTED)
text(CXP + 22, QY + 312, "Costs 0.12 pp of mAP@0.5 in-domain but is retained "
     "for its cross-domain contribution.", 10.5, "normal", MUTED)

out.append('</svg>')

HERE = os.path.dirname(os.path.abspath(__file__))
dest = os.path.join(HERE, "Fig03_RetinaNeXt_architecture.svg")
with open(dest, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("wrote", dest)
