#!/usr/bin/env python3
"""Build a 4-column comparison row for each (png, mp4) pair in this folder.

Layout per row: [ Input Image | frame_a | frame_b | frame_c ]
Labels on top:  "Input Image" over col 0, "Generated Result" over cols 1-3.
Skips the comparison_* pair.
"""
import os
import glob
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "rows")
os.makedirs(OUT_DIR, exist_ok=True)

CELL = 512          # square cell size
GAP = 10            # gap between cells
MARGIN = 10         # outer margin
LABEL_H = 70        # height of the label band
BG = (255, 255, 255)
FG = (20, 20, 20)
FRAME_FRACTIONS = [1/6, 1/2, 5/6]   # turntable angles to sample

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font = ImageFont.truetype(FONT_PATH, 40)


def flatten(img, bg=BG):
    """Composite any transparency onto a solid background."""
    img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, bg + (255,))
    canvas.alpha_composite(img)
    return canvas.convert("RGB")


def fit_square(img, size=CELL, bg=BG):
    """Resize preserving aspect ratio, letterbox onto a square canvas."""
    img = flatten(img, bg)
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def cover_square(img, size=CELL, bg=BG):
    """Center-crop to a square, then rescale to fill the cell (no letterbox)."""
    img = flatten(img, bg)
    side = min(img.width, img.height)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def grab_frames(mp4, fractions):
    """Extract frames at given fractions of the clip using ffprobe+ffmpeg."""
    nb = int(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "csv=p=0", mp4,
    ]).decode().strip())
    frames = []
    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(fractions):
            idx = min(nb - 1, max(0, int(round(fr * (nb - 1)))))
            out = os.path.join(td, f"f{i}.png")
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
                "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", out,
            ], check=True)
            frames.append(Image.open(out).copy())
    return frames


def centered_text(draw, x_center, y, text):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((x_center - w / 2, y), text, fill=FG, font=font)


def build_row(name, png, mp4):
    cells = [cover_square(Image.open(png))]
    cells += [fit_square(f) for f in grab_frames(mp4, FRAME_FRACTIONS)]

    n = len(cells)  # 4
    W = MARGIN * 2 + CELL * n + GAP * (n - 1)
    H = MARGIN + LABEL_H + CELL + MARGIN
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    xs = [MARGIN + i * (CELL + GAP) for i in range(n)]
    y_img = MARGIN + LABEL_H
    for x, cell in zip(xs, cells):
        canvas.paste(cell, (x, y_img))

    # labels
    label_y = MARGIN + (LABEL_H - 40) // 2
    centered_text(draw, xs[0] + CELL / 2, label_y, "Input Image")
    gen_left = xs[1]
    gen_right = xs[3] + CELL
    centered_text(draw, (gen_left + gen_right) / 2, label_y, "Generated Result")

    out = os.path.join(OUT_DIR, f"{name}_row.png")
    canvas.save(out)
    print(f"  -> {out}  ({W}x{H})")


def main():
    pngs = sorted(glob.glob(os.path.join(HERE, "*.png")))
    for png in pngs:
        name = os.path.splitext(os.path.basename(png))[0]
        if name.startswith("comparison_"):
            continue
        mp4 = os.path.join(HERE, name + ".mp4")
        if not os.path.exists(mp4):
            print(f"[skip] {name}: no matching mp4")
            continue
        print(f"[row] {name}")
        build_row(name, png, mp4)


if __name__ == "__main__":
    main()
