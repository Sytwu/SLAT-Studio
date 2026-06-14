# SLAT-Studio — Project Page

A static project page for **SLAT-Studio: Training-Free Downstream 3D Tasks on Structured Latents**.
Open `index.html` directly, or serve the folder (e.g. `python -m http.server` from here) — it's a
plain static site, deployable as-is to GitHub Pages.

## Contents
- `index.html` — the page (teaser, abstract, method, one section per task, BibTeX).
- `static/videos/` — the result media shown on the page (synchronized turntables):
  - `teaser.mp4` — one input asset → restyle / edit / inpaint, side by side.
  - `edit.mp4`, `inpaint.mp4`, `morph.mp4` — per-task comparison clips.
  - `turntable_source.mp4`, `turntable_restyle.mp4` — input vs. restyled chest (style-transfer section).
- `static/images/` — the matching still figures (kept on disk; `teaser.png` is also the social-preview image):
  - `teaser.png`, `carousel_{restyle,edit,inpaint,morph}.png` — still versions of the comparisons.
  - `hero_*.png` — single-view stills of each asset (intermediates used to build the composites).
  - `method_pipeline.png` — the TRELLIS SLAT pipeline figure (shown as an image on the page).

## Regenerating the figures
The result figures are rendered from the app's `.ply` outputs in `../outputs/` by:

```bash
# from the repo root, with the trellis env on PATH (see docs/STATUS.md)
export PATH=/home/cookies/miniconda3/envs/trellis/bin:/usr/local/cuda-11.8/bin:$PATH
export PYTHONPATH=$PWD/third_party/TRELLIS ATTN_BACKEND=xformers SPCONV_ALGO=native CUDA_VISIBLE_DEVICES=1
python scripts/make_page_figures.py
```

## Still to fill in (left as placeholders on purpose)
Author names, affiliation/venue, social-preview/Twitter handles, and the deployed page URL —
search `index.html` for `TODO`. The favicon under `static/images/favicon.ico` is still the
template default and should be replaced.

---
Built from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
(adapted from [Nerfies](https://nerfies.github.io)); licensed CC BY-SA 4.0.
