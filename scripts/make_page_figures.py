"""Render project-page figures from the app's .ply outputs.

Loads each gaussian-splat .ply (transform=None — avoids the vendored load_ply
transform-branch bug, and keeps every asset in one consistent frame), renders a
fixed 3/4 hero view + a turntable, and composites labeled comparison figures
with PIL. No flow models are loaded — pure decode-free rendering.
"""
import os
import numpy as np
import torch
import imageio.v2 as imageio
import utils3d
from PIL import Image, ImageDraw, ImageFont
from trellis.representations import Gaussian
from trellis.utils import render_utils as ru

OUT = "outputs"
DST = "project-page/static/images"
VID = "project-page/static/videos"
os.makedirs(DST, exist_ok=True)
os.makedirs(VID, exist_ok=True)

# The app saves .ply in a frame where the asset's vertical axis is Y, but TRELLIS's stock
# orbit cameras use up=[0,0,1] — so rendering the saved .ply with them makes the chest lie
# on its back. We build our own cameras that orbit horizontally around the vertical (Y)
# axis. The chest's true "up" is -Y in this frame (with +Y up it renders upside-down), so
# up = [0,-1,0] and the camera is elevated along -Y.
# yaw=pi/2 puts the lava-edited / carved face toward the camera, so the single hero view
# reveals every task's modification (all four wooden faces look alike, so it still reads as
# "facing the camera").
HERO_YAW, HERO_PITCH, R, FOV = np.pi / 2, 0.2, 2.0, 40
BG = (1.0, 1.0, 1.0)
RES = 768
UP = torch.tensor([0.0, -1.0, 0.0]).cuda()
TGT = torch.tensor([0.0, 0.0, 0.0]).cuda()

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)


def yup_cams(yaws, pitch, r=R, fov=FOV):
    """Cameras orbiting horizontally around the +Y (vertical) axis, looking at the origin."""
    f = torch.deg2rad(torch.tensor(float(fov))).cuda()
    ext, intr = [], []
    for yaw in yaws:
        y = torch.tensor(float(yaw)).cuda()
        p = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(y) * torch.cos(p),
            -torch.sin(p),  # elevate along the asset's true up (-Y)
            torch.cos(y) * torch.cos(p),
        ]).cuda() * r
        ext.append(utils3d.torch.extrinsics_look_at(orig, TGT, UP))
        intr.append(utils3d.torch.intrinsics_from_fov_xy(f, f))
    return ext, intr


def load(name):
    gs = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], sh_degree=0, device="cuda")
    gs.load_ply(os.path.join(OUT, name), transform=None)
    return gs


def hero(gs, res=RES):
    ext, intr = yup_cams([HERO_YAW], HERO_PITCH)
    r = ru.render_frames(gs, ext, intr, {"resolution": res, "bg_color": BG}, verbose=False)
    return r["color"][0]


def turntable(gs, path, nframes=120, res=512):
    yaws = (HERO_YAW + torch.linspace(0, 2 * np.pi, nframes)).tolist()
    ext, intr = yup_cams(yaws, HERO_PITCH)
    r = ru.render_frames(gs, ext, intr, {"resolution": res, "bg_color": BG}, verbose=False)
    imageio.mimsave(path, r["color"], fps=30, quality=8)
    print("wrote", path)


VFRAMES = 90  # frames per full revolution for the comparison videos


def tt_frames(name, res):
    """One full horizontal revolution of an asset, as a list of numpy frames."""
    gs = load(name)
    yaws = (HERO_YAW + torch.linspace(0, 2 * np.pi, VFRAMES)).tolist()
    ext, intr = yup_cams(yaws, HERO_PITCH)
    r = ru.render_frames(gs, ext, intr, {"resolution": res, "bg_color": BG}, verbose=False)
    del gs
    torch.cuda.empty_cache()
    return r["color"]


def sbs_video(specs, path, res=384, gap=18):
    """specs: list of (ply_name, caption). Synchronized turntables, labeled, side by side."""
    cols = [(tt_frames(n, res), cap) for n, cap in specs]
    out = []
    for i in range(VFRAMES):
        row = [label(col[i], cap) for col, cap in cols]
        out.append(np.array(hstack(row, gap=gap)))
    imageio.mimsave(path, out, fps=30, quality=8)
    print("wrote", path)


def morph_video(path, res=420):
    """Rotate while stepping through the morph frames f0..f4, then ping-pong back for a clean loop."""
    yaws_all = (HERO_YAW + torch.linspace(0, 2 * np.pi, VFRAMES)).tolist()
    per = VFRAMES // 5
    out = []
    for i in range(5):
        gs = load(f"my_asset_morph_f{i}.ply")
        lo, hi = i * per, (VFRAMES if i == 4 else (i + 1) * per)
        ext, intr = yup_cams(yaws_all[lo:hi], HERO_PITCH)
        r = ru.render_frames(gs, ext, intr, {"resolution": res, "bg_color": BG}, verbose=False)
        out.extend(np.array(label(f, f"t = {i/4:.2f}")) for f in r["color"])
        del gs
        torch.cuda.empty_cache()
    out = out + out[::-1]
    imageio.mimsave(path, out, fps=30, quality=8)
    print("wrote", path)


def label(img_np, text, pad=14):
    """Return a PIL image of img_np with a caption bar under it."""
    img = Image.fromarray(img_np)
    w, h = img.size
    bar = 56
    canvas = Image.new("RGB", (w, h + bar), (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    tw = d.textbbox((0, 0), text, font=FONT)[2]
    d.text(((w - tw) / 2, h + (bar - 40) / 2), text, font=FONT, fill=(20, 20, 20))
    return canvas


def hstack(imgs, gap=18):
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


# ---- render every asset's hero still -----------------------------------------
ASSETS = {
    "source":    "my_asset.ply",
    "restyled":  "my_asset_restyled.ply",
    "edited":    "my_asset_edited.ply",
    "region":    "my_asset_region.ply",          # green-tinted edit region preview
    "hole":      "my_asset_completed_carved.ply",  # carved (before inpaint)
    "completed": "my_asset_completed.ply",       # after inpaint
}
heroes = {}
for key, fn in ASSETS.items():
    gs = load(fn)
    img = hero(gs)
    heroes[key] = img
    imageio.imwrite(os.path.join(DST, f"hero_{key}.png"), img)
    print("hero", key, img.shape)
    del gs
    torch.cuda.empty_cache()

# morph frames f0..f4
morph_imgs = []
for i in range(5):
    gs = load(f"my_asset_morph_f{i}.ply")
    img = hero(gs, res=512)
    morph_imgs.append(img)
    del gs
    torch.cuda.empty_cache()

# ---- composites --------------------------------------------------------------
# 1) teaser: one input -> four downstream edits
teaser = hstack([
    label(heroes["source"], "Input asset"),
    label(heroes["restyled"], "Restyle (gold)"),
    label(heroes["edited"], "Edit (lava region)"),
    label(heroes["completed"], "Inpaint / complete"),
])
teaser.save(os.path.join(DST, "teaser.png"))
print("wrote teaser.png")

# 2) restyle pair
hstack([label(heroes["source"], "Input (wooden)"),
        label(heroes["restyled"], "Restyled (solid gold)")]).save(
    os.path.join(DST, "carousel_restyle.png"))

# 3) edit triple: input | region | edited
hstack([label(heroes["source"], "Input"),
        label(heroes["region"], "Selected region"),
        label(heroes["edited"], 'Edited ("lava")')]).save(
    os.path.join(DST, "carousel_edit.png"))

# 4) inpaint triple: input | carved hole | completed
hstack([label(heroes["source"], "Input"),
        label(heroes["hole"], "Region carved"),
        label(heroes["completed"], "Completed")]).save(
    os.path.join(DST, "carousel_inpaint.png"))

# 5) morph strip f0..f4
strip = hstack([label(m, f"t = {i/4:.2f}") for i, m in enumerate(morph_imgs)], gap=10)
strip.save(os.path.join(DST, "carousel_morph.png"))
print("wrote carousel_*.png")

# ---- turntable videos --------------------------------------------------------
turntable(load("my_asset.ply"), os.path.join(VID, "turntable_source.mp4"))
turntable(load("my_asset_restyled.ply"), os.path.join(VID, "turntable_restyle.mp4"))

# ---- comparison videos (the page shows these in place of the still figures) --
sbs_video([("my_asset.ply", "Input asset"),
           ("my_asset_restyled.ply", "Restyle (gold)"),
           ("my_asset_edited.ply", "Edit (lava region)"),
           ("my_asset_completed.ply", "Inpaint / complete")],
          os.path.join(VID, "teaser.mp4"))
sbs_video([("my_asset.ply", "Input"),
           ("my_asset_region.ply", "Selected region"),
           ("my_asset_edited.ply", 'Edited ("lava")')],
          os.path.join(VID, "edit.mp4"))
sbs_video([("my_asset.ply", "Input"),
           ("my_asset_completed_carved.ply", "Region carved"),
           ("my_asset_completed.ply", "Completed")],
          os.path.join(VID, "inpaint.mp4"))
morph_video(os.path.join(VID, "morph.mp4"))
print("done")
