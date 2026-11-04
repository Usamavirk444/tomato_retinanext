"""
make_arch_svg.py
────────────────
Figure 2 — the four compared architectures, 2 x 2 panels.

Layout follows the reference composition: centred panel titles with the
parameter count and throughput on a second line, isometric 3D feature
blocks, navy backbones, golden necks, pale blue heads, and a red border on
the proposed model.

Fixed relative to the reference image:
    "Six featuure scales"                  typo
    duplicated "Six feature scales" label  drawn once
    duplicated "Input 640 x 640" labels    drawn once per panel
    missing "C5 2048 ch" block             restored in panel C
    missing C5 / CBAM labels in panel D    restored

Block geometry carries information:
    block HEIGHT  proportional to spatial resolution (sqrt scale)
    block WIDTH   proportional to channel count      (sqrt scale)
so ConvNeXt-Tiny's 192/384/768 against ResNet-50's 512/1024/2048 is visible.

Verified against the project:
    input 640x640              configs/experiment.yaml:45
    ConvNeXt C3/C4/C5          192/384/768   convnext_backbone.py
    ResNet-50 C3/C4/C5         512/1024/2048
    BiFPN 3 repeats, 256 ch    bifpn.py, retinanet_hybrid.py
    CBAM at five levels        CBAMNeck(num_levels=5)
    params / FPS               outputs/results/fps_benchmark.csv

Usage:
    python make_arch_svg.py       -> Fig02_four_architectures.svg
"""

import math
import os

# ── palette, shared with make_fig03_svg.py ───────────────────────────
NAVY   = "#1F2A44"; NAVY_T  = "#36425F"; NAVY_S  = "#121829"
GOLD   = "#DFB13C"; GOLD_T  = "#EFC866"; GOLD_S  = "#A07C16"
BLUE   = "#C5D9F1"; BLUE_T  = "#DEEAF7"; BLUE_S  = "#7FA6CF"
CREAM  = "#F6EDC8"; CREAM_T = "#FBF6E2"; CREAM_S = "#C9A94A"
PINK   = "#F4CCCC"; PINK_T  = "#FAE3E3"; PINK_S  = "#C0504D"
WHITE  = "#FFFFFF"; WHITE_T = "#FFFFFF"; WHITE_S = "#9A9A9A"
BG     = "#F7F8FA"; PANEL   = "#FFFFFF"; EDGE = "#4472C4"
ACCENT = "#C0504D"; TEXT    = "#111111"; MUTED = "#555555"
GY     = "#3C3C3C"

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


def slab(spatial, channels, s=0.50, c=1.35):
    return (max(13, math.sqrt(channels) * 1.05 * c),
            max(24, math.sqrt(spatial) * 11.0 * s))


def group(x, ym, spatial, channels, n, face, top, side, gap=7, s=0.50, c=1.35):
    w, h = slab(spatial, channels, s, c)
    for i in range(n):
        cuboid(x + i * (w + gap), ym - h / 2, w, h, face, top, side)
    return x + n * w + (n - 1) * gap, h


def arrow(x1, y, x2, colour=GY, marker="ahk", width=1.8):
    out.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2-6:.1f}" y2="{y:.1f}" '
               f'stroke="{colour}" stroke-width="{width}" marker-end="url(#{marker})"/>')


def panel(x, y, w, h, letter, title, meta, stroke=EDGE, sw=1.5):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
               f'fill="{PANEL}" stroke="{stroke}" stroke-width="{sw}"/>')
    text(x + w / 2, y + 34, f"{letter}. {title}", 17, "bold", TEXT, "middle")
    text(x + w / 2, y + 58, meta, 15, "normal", MUTED, "middle")


# ════════════════════════════════════════════════════════════════════
W, H = 1320, 830
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}">')
out.append('<defs>'
           f'<marker id="ahk" markerWidth="7" markerHeight="7" refX="6" refY="3" '
           f'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="{GY}"/></marker>'
           f'<marker id="ahr" markerWidth="7" markerHeight="7" refX="6" refY="3" '
           f'orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="{ACCENT}"/></marker>'
           '</defs>')
out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

text(W / 2, 44, "COMPARATIVE ARCHITECTURES FOR TOMATO LEAF DISEASE DETECTION",
     21, "bold", TEXT, "middle")

PW, PH = 628, 358
PX1, PX2, PY1, PY2 = 18, 674, 72, 442

# ── A: SSDLite ───────────────────────────────────────────────────────
panel(PX1, PY1, PW, PH, "A", "SSDLite with MobileNetV3-Large",
      "2.3 M parameters, 48.4 FPS")
ym = PY1 + 200
cuboid(PX1 + 36, ym - 32, 62, 64, WHITE, WHITE_T, WHITE_S)
text(PX1 + 67, ym - 58, "Input", 11.5, "bold", TEXT, "middle")
text(PX1 + 67, ym - 44, "640 x 640", 10.5, "normal", MUTED, "middle")
arrow(PX1 + 112, ym, PX1 + 142)
cuboid(PX1 + 142, ym - 56, 48, 112, NAVY, NAVY_T, NAVY_S)
text(PX1 + 166, ym + 76, "MobileNetV3-Large", 11.5, "bold", TEXT, "middle")
text(PX1 + 166, ym + 90, "backbone", 11.5, "bold", TEXT, "middle")
text(PX1 + 166, ym + 105, "inverted residual + SE", 10, "normal", MUTED, "middle")
arrow(PX1 + 206, ym, PX1 + 238)
gx, gh = group(PX1 + 238, ym, 320, 320, 6, GOLD, GOLD_T, GOLD_S, gap=6, c=0.80)
gm = (PX1 + 238 + gx) / 2
text(gm, ym - gh / 2 - 36, "Six feature scales", 11.5, "bold", TEXT, "middle")
text(gm, ym - gh / 2 - 22, "hierarchical downsampling", 10, "normal", MUTED, "middle")
arrow(gx + DEPTH + 4, ym, gx + DEPTH + 36)
hx = gx + DEPTH + 36
cuboid(hx, ym - 44, 54, 88, BLUE, BLUE_T, BLUE_S)
text(hx + 27, ym + 66, "SSDLite heads", 11.5, "bold", TEXT, "middle")
text(hx + 27, ym + 80, "depthwise separable", 10, "normal", MUTED, "middle")
arrow(hx + 54 + DEPTH + 4, ym, hx + 54 + DEPTH + 34)
text(hx + 54 + DEPTH + 40, ym - 4, "9-class", 11, "normal")
text(hx + 54 + DEPTH + 40, ym + 11, "detections", 11, "normal")
text(PX1 + 36, PY1 + PH - 22, "The only model here with no feature-pyramid neck.",
     11, "normal", ACCENT)

# ── B: YOLOv8n ───────────────────────────────────────────────────────
panel(PX2, PY1, PW, PH, "B", "YOLOv8n with CSPDarknet-C2f",
      "3.0 M parameters, 83.5 FPS")
cuboid(PX2 + 36, ym - 32, 62, 64, WHITE, WHITE_T, WHITE_S)
text(PX2 + 67, ym - 58, "Input", 11.5, "bold", TEXT, "middle")
text(PX2 + 67, ym - 44, "640 x 640", 10.5, "normal", MUTED, "middle")
arrow(PX2 + 112, ym, PX2 + 144)
bx, bh = group(PX2 + 144, ym, 320, 256, 3, NAVY, NAVY_T, NAVY_S, gap=7, c=1.05)
bm = (PX2 + 144 + bx) / 2
text(bm, ym + bh / 2 + 28, "CSPDarknet", 11.5, "bold", TEXT, "middle")
text(bm, ym + bh / 2 + 42, "C2f", 11.5, "bold", TEXT, "middle")
text(bm, ym + bh / 2 + 57, "split / merge gradients", 10, "normal", MUTED, "middle")
arrow(bx + DEPTH + 4, ym, bx + DEPTH + 36)
nx, nh = group(bx + DEPTH + 36, ym, 250, 256, 3, GOLD, GOLD_T, GOLD_S, gap=7, c=1.05)
nm = (bx + DEPTH + 36 + nx) / 2
text(nm, ym + nh / 2 + 28, "PAN neck", 11.5, "bold", TEXT, "middle")
text(nm, ym + nh / 2 + 42, "path aggregation", 10, "normal", MUTED, "middle")
arrow(nx + DEPTH + 4, ym, nx + DEPTH + 36)
hx = nx + DEPTH + 36
cuboid(hx, ym - 44, 54, 88, BLUE, BLUE_T, BLUE_S)
text(hx + 27, ym + 66, "Decoupled head", 11.5, "bold", TEXT, "middle")
text(hx + 27, ym + 80, "anchor-free", 10, "normal", MUTED, "middle")
arrow(hx + 54 + DEPTH + 4, ym, hx + 54 + DEPTH + 34)
text(hx + 54 + DEPTH + 40, ym - 4, "9-class", 11, "normal")
text(hx + 54 + DEPTH + 40, ym + 11, "detections", 11, "normal")
text(PX2 + 36, PY1 + PH - 22, "Trained through the Ultralytics pipeline, which "
     "adds mosaic, copy-paste and mixup augmentation the other three do not.",
     10, "normal", MUTED)

# ── C: RetinaNet ─────────────────────────────────────────────────────
panel(PX1, PY2, PW, PH, "C", "RetinaNet with ResNet-50",
      "36.5 M parameters, 32.3 FPS")
ym2 = PY2 + 200
cuboid(PX1 + 36, ym2 - 32, 62, 64, WHITE, WHITE_T, WHITE_S)
text(PX1 + 67, ym2 + 62, "Input", 11.5, "bold", TEXT, "middle")
text(PX1 + 67, ym2 + 76, "640 x 640", 10.5, "normal", MUTED, "middle")
arrow(PX1 + 112, ym2, PX1 + 140)
bx, last_end = PX1 + 140, PX1 + 140
for sp, ch, nm_ in [(80, 512, "C3"), (40, 1024, "C4"), (20, 2048, "C5")]:
    w, h = slab(sp, ch, 0.50, 0.95)
    cuboid(bx, ym2 - h / 2, w, h, NAVY, NAVY_T, NAVY_S)
    text(bx + w / 2, ym2 - h / 2 - 28, nm_, 11.5, "bold", TEXT, "middle")
    text(bx + w / 2, ym2 - h / 2 - 14, f"{ch} ch", 10, "normal", MUTED, "middle")
    last_end = bx + w + DEPTH
    bx += 56
text(PX1 + 140 + 56, ym2 + 76, "ResNet-50", 11.5, "bold", TEXT, "middle")
arrow(last_end + 4, ym2, last_end + 32)
fx, fh = group(last_end + 32, ym2, 250, 256, 5, GOLD, GOLD_T, GOLD_S, gap=6, c=0.95)
fm = (last_end + 32 + fx) / 2
text(fm, ym2 - fh / 2 - 36, "FPN P3-P7", 11.5, "bold", TEXT, "middle")
text(fm, ym2 - fh / 2 - 22, "top-down only, 256 ch", 10, "normal", MUTED, "middle")
arrow(fx + DEPTH + 4, ym2, fx + DEPTH + 34)
hx = fx + DEPTH + 34
cuboid(hx, ym2 - 46, 56, 92, BLUE, BLUE_T, BLUE_S)
text(hx + 28, ym2 + 68, "Shared subnets", 11.5, "bold", TEXT, "middle")
text(hx + 28, ym2 + 82, "focal loss, alpha 0.25, gamma 2.0", 10, "normal",
     MUTED, "middle")
arrow(hx + 56 + DEPTH + 4, ym2, hx + 56 + DEPTH + 32)
text(hx + 56 + DEPTH + 38, ym2 - 4, "9-class", 11, "normal")
text(hx + 56 + DEPTH + 38, ym2 + 11, "detections", 11, "normal")
text(PX1 + 36, PY2 + PH - 22, "Shares its head, anchors and loss with RetinaNeXt, "
     "so the comparison isolates backbone, neck and attention.", 10, "normal", MUTED)

# ── D: RetinaNeXt ────────────────────────────────────────────────────
panel(PX2, PY2, PW, PH, "D", "RetinaNeXt with ConvNeXt-Tiny - proposed",
      "35.2 M parameters, 26.1 FPS", ACCENT, 2.2)
cuboid(PX2 + 32, ym2 - 32, 52, 64, PINK, PINK_T, PINK_S)
text(PX2 + 58, ym2 + 62, "Input", 11.5, "bold", TEXT, "middle")
text(PX2 + 58, ym2 + 76, "640 x 640", 10.5, "normal", MUTED, "middle")
arrow(PX2 + 98, ym2, PX2 + 124, ACCENT, "ahr")
bx, last_end = PX2 + 124, PX2 + 124
for sp, ch, nm_ in [(80, 192, "C3"), (40, 384, "C4"), (20, 768, "C5")]:
    w, h = slab(sp, ch, 0.50, 0.95)
    cuboid(bx, ym2 - h / 2, w, h, NAVY, NAVY_T, NAVY_S)
    text(bx + w / 2, ym2 - h / 2 - 28, nm_, 11.5, "bold", TEXT, "middle")
    text(bx + w / 2, ym2 - h / 2 - 14, f"{ch} ch", 10, "normal", MUTED, "middle")
    last_end = bx + w + DEPTH
    bx += 44
text(PX2 + 124 + 44, ym2 + 76, "ConvNeXt-Tiny", 11.5, "bold", TEXT, "middle")
arrow(last_end + 4, ym2, last_end + 28, ACCENT, "ahr")
fx, fh = group(last_end + 28, ym2, 250, 256, 3, GOLD, GOLD_T, GOLD_S, gap=6, c=0.85)
fm = (last_end + 28 + fx) / 2
text(fm, ym2 - fh / 2 - 36, "BiFPN x3", 11.5, "bold", TEXT, "middle")
text(fm, ym2 + fh / 2 + 24, "C3-C5 only", 10, "normal", MUTED, "middle")
arrow(fx + DEPTH + 4, ym2, fx + DEPTH + 26, ACCENT, "ahr")
cx, ch_ = group(fx + DEPTH + 26, ym2, 250, 256, 5, CREAM, CREAM_T, CREAM_S,
                gap=5, c=0.45)
cm = (fx + DEPTH + 26 + cx) / 2
text(cm, ym2 - ch_ / 2 - 36, "CBAM", 11.5, "bold", TEXT, "middle")
text(cm, ym2 - ch_ / 2 - 22, "P3-P7", 10, "normal", MUTED, "middle")
arrow(cx + DEPTH + 4, ym2, cx + DEPTH + 26, ACCENT, "ahr")
hx = cx + DEPTH + 26
cuboid(hx, ym2 - 46, 48, 92, BLUE, BLUE_T, BLUE_S)
text(hx + 24, ym2 + 68, "Shared subnets", 11.5, "bold", TEXT, "middle")
text(hx + 24, ym2 + 82, "identical to C", 10, "normal", MUTED, "middle")
arrow(hx + 48 + DEPTH + 4, ym2, hx + 48 + DEPTH + 28, ACCENT, "ahr")
text(hx + 48 + DEPTH + 34, ym2 - 4, "9-class", 11, "normal")
text(hx + 48 + DEPTH + 34, ym2 + 11, "detections", 11, "normal")
text(PX2 + 36, PY2 + PH - 22, "P6 and P7 are added after the BiFPN by strided "
     "convolution, not fused inside it.", 10.5, "normal", ACCENT)

out.append('</svg>')

HERE = os.path.dirname(os.path.abspath(__file__))
dest = os.path.join(HERE, "Fig02_four_architectures.svg")
with open(dest, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("wrote", dest)
