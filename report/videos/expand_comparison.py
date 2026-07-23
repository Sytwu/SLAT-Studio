#!/usr/bin/env python3
"""Expand comparison_*.png (3x4 input grid) into a 3x8 grid where each input
cell is followed by the matching generated-result frame from the video.

- Input grid order is preserved.
- The video cells are score-sorted, so they are matched to inputs by label.
- The burned-in black/white label bars are cropped out; a single perturbation
  label is drawn above each input+result pair.
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = os.path.join(HERE, "comparison_typical_creature_dragon.png")
FRAME = "/tmp/frame60.png"
OUT = os.path.join(HERE, "comparison_typical_creature_dragon_expanded.png")

# --- PNG (input) grid geometry (detected) ---
PNG_COL_X = [8, 735, 1462, 2189]   # content left of each column
PNG_CW = 720
PNG_ROW_Y = [38, 795, 1552]        # content top of each row (below label bar)
PNG_RH = 719

# --- Video frame grid geometry ---
VID_COLS, VID_ROWS = 4, 3
VID_LABEL_H = 30                    # white text strip to crop from each cell top

# input grid layout (row-major), perturbation labels as on the black bars
PNG_LABELS = [
    ["clean", "occ 15%", "occ 30%", "blur σ5"],
    ["bg black", "bg natural", "jpeg q10", "rot 90°"],
    ["rot 180°", "downscale 64", "canny edges", "hflip"],
]
# where each label lives in the (score-sorted) video grid: (row, col)
VID_POS = {
    "clean": (0, 0), "bg black": (0, 1), "occ 30%": (0, 2), "bg natural": (0, 3),
    "occ 15%": (1, 0), "jpeg q10": (1, 1), "blur σ5": (1, 2), "hflip": (1, 3),
    "downscale 64": (2, 0), "rot 90°": (2, 1), "rot 180°": (2, 2), "canny edges": (2, 3),
}

# --- output layout ---
CELL = 340
GAP_IN = 6        # gap inside an input+result pair
GAP_PAIR = 30     # gap between pairs
MARGIN = 16
LABEL_H = 50
ROW_GAP = 18
BG = (255, 255, 255)
FG = (20, 20, 20)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)


def png_cell(png, r, c):
    x, y = PNG_COL_X[c], PNG_ROW_Y[r]
    return png.crop((x, y, x + PNG_CW, y + PNG_RH)).resize((CELL, CELL), Image.LANCZOS)


def vid_cell(frame, r, c):
    W, H = frame.size
    cw = W / VID_COLS
    x0 = round(c * cw)
    x1 = round((c + 1) * cw)
    y0 = round(r * H / VID_ROWS) + VID_LABEL_H
    y1 = round((r + 1) * H / VID_ROWS)
    return frame.crop((x0, y0, x1, y1)).resize((CELL, CELL), Image.LANCZOS)


def main():
    png = Image.open(PNG).convert("RGB")
    frame = Image.open(FRAME).convert("RGB")

    pair_w = 2 * CELL + GAP_IN
    W = 2 * MARGIN + 4 * pair_w + 3 * GAP_PAIR
    H = MARGIN + 3 * (LABEL_H + CELL) + 2 * ROW_GAP + MARGIN
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    pair_x = [MARGIN + p * (pair_w + GAP_PAIR) for p in range(4)]

    for r in range(3):
        y_label = MARGIN + r * (LABEL_H + CELL + ROW_GAP)
        y_img = y_label + LABEL_H
        for c in range(4):
            label = PNG_LABELS[r][c]
            inp = png_cell(png, r, c)
            vr, vc = VID_POS[label]
            res = vid_cell(frame, vr, vc)

            xin = pair_x[c]
            xres = xin + CELL + GAP_IN
            canvas.paste(inp, (xin, y_img))
            canvas.paste(res, (xres, y_img))

            # centered label over the pair
            bbox = draw.textbbox((0, 0), label, font=FONT)
            tw = bbox[2] - bbox[0]
            cx = xin + pair_w / 2
            draw.text((cx - tw / 2, y_label + (LABEL_H - 34) / 2), label,
                      fill=FG, font=FONT)

    canvas.save(OUT)
    print(f"saved {OUT}  ({W}x{H})")


if __name__ == "__main__":
    main()
