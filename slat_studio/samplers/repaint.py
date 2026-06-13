"""RePaint-style masked flow sampler for SLAT region editing.

Phase 3's core. We re-generate the latents *inside* a region (the "unknown" voxels)
conditioned on a new text prompt, while keeping the latents *outside* the region (the
"known" voxels) pinned to the source asset — RePaint (Lugmayr et al., 2022) adapted from
DDPM to TRELLIS's flow-matching sampler.

Why subclass instead of editing TRELLIS:
    The slat sampler TRELLIS ships is ``FlowEulerGuidanceIntervalSampler`` (CFG +
    guidance-interval). We subclass it so the classifier-free-guidance behaviour is reused
    verbatim (via the mixin's ``_inference_model``); we only override ``sample`` to inject
    the known latents at every Euler step. Nothing under ``third_party/TRELLIS`` is touched.

Flow-matching facts used (from ``FlowEulerSampler``):
    forward interpolation   x_t = (1 - t) * x_0 + s(t) * eps,   s(t) = sigma_min + (1-sigma_min)*t
    so the known x_0 can be re-noised to any t with fresh Gaussian eps. At t=0, s(0)=sigma_min
    (~1e-5), so the known region converges to x_0 up to a negligible jitter; the high-level
    editor additionally hard-composites the source latents back outside the mask to guarantee
    a *bit-exact* preservation there.

The mask is an ``[N, 1]`` float tensor (1 = regenerate, 0 = keep), row-aligned with the
SparseTensor feats. SparseTensor ``*`` with an ``[N,1]`` tensor broadcasts per-voxel over the
8 channels (the ``[1,C]`` batch-broadcast path raises and falls back to a plain elementwise
mul), which is exactly the per-voxel gating we want.
"""
from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

from trellis.pipelines.samplers import FlowEulerGuidanceIntervalSampler


class RepaintFlowSampler(FlowEulerGuidanceIntervalSampler):
    """Masked flow sampler: resample inside a voxel mask, keep known latents outside.

    Same CFG/guidance-interval behaviour as the base slat sampler; ``sample`` additionally
    takes the known clean latents ``x_0_known`` (a SparseTensor, in the sampler's *normalized*
    space, same coords/ordering as ``noise``) and a per-voxel ``mask`` ``[N,1]``.
    """

    def _s(self, t: float) -> float:
        return self.sigma_min + (1 - self.sigma_min) * t

    def _renoise_known(self, x_0_known, t: float):
        """Forward-diffuse the known x_0 to time t with fresh noise: x_t^known."""
        eps = torch.randn_like(x_0_known.feats)
        return x_0_known * (1 - t) + x_0_known.replace(eps) * self._s(t)

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        x_0_known,
        mask,
        steps: int = 25,
        rescale_t: float = 3.0,
        cfg_strength: float = 7.5,
        cfg_interval: Tuple[float, float] = (0.5, 0.95),
        resample: int = 1,
        verbose: bool = True,
        **kwargs,
    ):
        """RePaint masked sampling.

        Args:
            model: the slat flow model.
            noise: initial noise SparseTensor (same coords/ordering as ``x_0_known``).
            cond / neg_cond: text conditioning (positive / null), as in the base sampler.
            x_0_known: clean source latents in *normalized* space (SparseTensor).
            mask: ``[N,1]`` float on the same device — 1 inside the edit region, 0 outside.
            resample: RePaint jump-back count per step (1 = disabled). >1 alternates
                denoise(t->t_prev) / renoise(t_prev->t) to harmonize the boundary.
            (steps/rescale_t/cfg_*: as in ``FlowEulerGuidanceIntervalSampler``.)
        """
        keep = 1 - mask
        inject = dict(neg_cond=neg_cond, cfg_strength=cfg_strength,
                      cfg_interval=cfg_interval, **kwargs)

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="RePaint", disable=not verbose):
            for u in range(resample):
                out = self.sample_once(model, sample, t, t_prev, cond, **inject)
                # unknown region follows the model; known region is the re-noised source
                x_prev_known = self._renoise_known(x_0_known, t_prev)
                sample = out.pred_x_prev * mask + x_prev_known * keep

                if u < resample - 1 and t_prev > 0:
                    # jump back t_prev -> t (RePaint resampling) using the x_0 estimate
                    x0_est = out.pred_x_0 * mask + x_0_known * keep
                    eps = torch.randn_like(x_0_known.feats)
                    sample = x0_est * (1 - t) + x_0_known.replace(eps) * self._s(t)
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class DenseRepaintFlowSampler(FlowEulerGuidanceIntervalSampler):
    """RePaint on TRELLIS's *stage-1* sparse-structure flow (dense latent grid).

    The structure model samples a dense latent ``[B, C, r, r, r]`` (``r=16``) which the
    sparse-structure decoder upsamples to a ``64**3`` occupancy. To *inpaint geometry* we
    RePaint that dense latent: regenerate the latent cells covering the hole while pinning the
    known cells to the encoded latent of the surrounding (holed) occupancy.

    Identical RePaint math to :class:`RepaintFlowSampler`, but on plain dense tensors instead
    of SparseTensors. ``x_0_known`` is the SS encoder output of the known occupancy and ``mask``
    is a dense ``[1,1,r,r,r]`` float (1 = regenerate, 0 = keep), broadcast over channels.
    The SS stage has no latent normalization (the flow trains directly on encoder outputs), so
    ``x_0_known`` is used as-is.
    """

    def _s(self, t: float) -> float:
        return self.sigma_min + (1 - self.sigma_min) * t

    def _renoise_known(self, x_0_known, t: float):
        eps = torch.randn_like(x_0_known)
        return x_0_known * (1 - t) + eps * self._s(t)

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        x_0_known,
        mask,
        steps: int = 25,
        rescale_t: float = 3.0,
        cfg_strength: float = 7.5,
        cfg_interval: Tuple[float, float] = (0.5, 0.95),
        resample: int = 1,
        verbose: bool = True,
        **kwargs,
    ):
        """Masked dense sampling.

        Args:
            model: the sparse-structure flow model.
            noise: initial dense noise ``[1,C,r,r,r]``.
            cond / neg_cond: text conditioning (positive / null).
            x_0_known: encoded known occupancy latent ``[1,C,r,r,r]`` (SS-latent space).
            mask: ``[1,1,r,r,r]`` float — 1 inside the hole (regenerate), 0 outside (keep).
            resample: RePaint jump-back count per step (1 = off; >1 harmonizes the boundary).
        """
        keep = 1 - mask
        inject = dict(neg_cond=neg_cond, cfg_strength=cfg_strength,
                      cfg_interval=cfg_interval, **kwargs)

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="RePaint-SS", disable=not verbose):
            for u in range(resample):
                out = self.sample_once(model, sample, t, t_prev, cond, **inject)
                x_prev_known = self._renoise_known(x_0_known, t_prev)
                sample = out.pred_x_prev * mask + x_prev_known * keep

                if u < resample - 1 and t_prev > 0:
                    x0_est = out.pred_x_0 * mask + x_0_known * keep
                    eps = torch.randn_like(x_0_known)
                    sample = x0_est * (1 - t) + eps * self._s(t)
            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


def bbox_mask(coords, bbox, device="cuda"):
    """Per-voxel mask for a voxel-index bounding box (inclusive lower, exclusive upper).

    Args:
        coords: ``[N,4]`` (batch,x,y,z) or ``[N,3]`` (x,y,z) int voxel indices — pass the
            SparseTensor's own ``.coords`` so the mask is row-aligned with its feats.
        bbox: ``(x0,y0,z0, x1,y1,z1)`` in voxel units; voxel i is inside iff
            ``x0<=x<x1`` and likewise for y,z.

    Returns:
        ``[N,1]`` float tensor on ``device`` — 1.0 inside the box, 0.0 outside.
    """
    c = coords.detach().long().cpu()
    if c.shape[1] == 4:
        c = c[:, 1:]
    x0, y0, z0, x1, y1, z1 = bbox
    inside = (
        (c[:, 0] >= x0) & (c[:, 0] < x1) &
        (c[:, 1] >= y0) & (c[:, 1] < y1) &
        (c[:, 2] >= z0) & (c[:, 2] < z1)
    )
    return inside.float().unsqueeze(1).to(device)
