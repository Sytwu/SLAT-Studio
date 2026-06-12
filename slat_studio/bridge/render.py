"""Multiview render of an input 3DGS for the encoding bridge.

Thin wrapper over ``trellis.utils.render_utils.render_multiview``. We deliberately reuse
TRELLIS's own renderer + camera generator so the returned ``extrinsics``/``intrinsics`` are
in the exact CV convention that ``utils3d.torch.project_cv`` consumes during feature
extraction — rendering and projection then share one consistent camera set (no Blender
transforms.json round-trip, no coordinate-flip bookkeeping).
"""


def render_multiview(gaussian, nviews=150, resolution=512):
    """Render ``nviews`` sphere views of a Gaussian asset on a black background.

    Returns:
        ``(colors, extrinsics, intrinsics)`` where ``colors`` is ``[V,res,res,3]`` uint8
        and the cameras are per-view CV matrices aligned with the renders.
    """
    from trellis.utils import render_utils
    return render_utils.render_multiview(gaussian, resolution=resolution, nviews=nviews)
