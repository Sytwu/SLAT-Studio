"""SLAT-Studio: downstream 3D tasks on top of TRELLIS structured latents (SLAT).

This package only *imports* from `trellis.*` (vendored unmodified as a git submodule
under `third_party/TRELLIS`). It never modifies the submodule; custom behaviour is
added by subclassing/composing TRELLIS classes.

Submodules
----------
bridge    : encode an existing 3DGS / mesh into SLAT.
editing   : region / local 3D editing.
style     : style transfer & material/appearance alteration.
morph     : SLAT interpolation & morphing.
inpainting: training-free 3D completion of a holed SLAT (two-stage RePaint).
io        : load/save SLAT, 3DGS (.ply), mesh, GLB/OBJ.
pipelines : high-level user-facing entry points.
samplers  : custom flow samplers (subclasses of TRELLIS samplers).
"""

__version__ = "0.0.0"
