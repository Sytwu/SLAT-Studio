"""SLAT morphing: interpolate between two assets' structured latents.

The hard part of morphing two SLATs is that they have *different* active-voxel sets
``{p_i}`` (different ``L`` and positions), so there is no row-to-row correspondence to
lerp. We solve this with a **structure union + occupancy schedule**:

  * Build the union of the two voxel grids. Each union voxel is *shared* (active in both),
    *A-only*, or *B-only*.
  * Latents: shared voxels are linearly interpolated ``(1-t) z_A + t z_B``; A-only and
    B-only voxels keep their native latent (there is no meaningful "empty" latent — an
    occupied voxel always decodes to content, so presence is controlled by occupancy, not
    by fading a latent to zero).
  * Occupancy (structure morph): each A-only voxel is assigned a stable random threshold
    ``tau`` and stays occupied while ``t < tau``; each B-only voxel appears once ``t > tau``.
    Shared voxels are always occupied. As ``t`` goes 0->1 this dithers A out and B in — a
    temporally-coherent dissolve (the same voxel transitions monotonically across ``t``,
    so there is no per-frame flicker). Endpoints are *exact*: ``t=0`` reproduces A, ``t=1``
    reproduces B (coords and latents), so the morph starts and ends on the real assets.

Every intermediate is a valid SLAT (every occupied voxel carries a real latent), so it
decodes with the stock GS/RF/mesh decoders. Pure composition of TRELLIS public methods +
``trellis.modules.sparse`` — nothing under ``third_party/TRELLIS`` is modified.

``harmonize_slat`` is an optional SDEdit pass (reuses the stock slat sampler's ``sample_once``)
that renoises the blended latent a little and denoises it, nudging an off-manifold lerp on
the *shared* voxels back toward the data manifold. Off by default.

Reference: MorphAny3D (arXiv:2601.00204).
"""
from typing import Optional, Sequence

__all__ = ["SlatMorpher", "morph_sequence", "harmonize_slat"]


def _lin(coords, res: int):
    """Flatten ``[N,4]`` (or ``[N,3]``) voxel coords to a 1-D key ``x*res^2 + y*res + z``."""
    c = coords[:, 1:].long() if coords.shape[1] == 4 else coords.long()
    return (c[:, 0] * res + c[:, 1]) * res + c[:, 2]


def _coords_from_keys(keys, res: int):
    """Inverse of :func:`_lin`: 1-D keys -> ``[N,4]`` ``[batch=0, x, y, z]`` int coords."""
    import torch
    z = keys % res
    y = (keys // res) % res
    x = keys // (res * res)
    batch = torch.zeros_like(x)
    return torch.stack([batch, x, y, z], dim=1).int()


class SlatMorpher:
    """Precompute the structure correspondence of two SLATs, then emit any intermediate.

    Build once (the union + aligned latents + dissolve thresholds are computed in ``__init__``),
    then call :meth:`at` for each fraction ``t`` — cheap, no model required.
    """

    def __init__(self, slat_A, slat_B, seed: int = 0, resolution: int = 64):
        import torch

        if slat_A.feats.device != slat_B.feats.device:
            raise ValueError("slat_A and slat_B must be on the same device")
        self.res = resolution
        self.device = slat_A.feats.device
        self.C = slat_A.feats.shape[1]

        keyA = _lin(slat_A.coords, resolution)
        keyB = _lin(slat_B.coords, resolution)
        union = torch.unique(torch.cat([keyA, keyB]))      # sorted ascending, unique
        self.union_keys = union
        N = union.shape[0]

        inA = torch.isin(union, keyA)
        inB = torch.isin(union, keyB)
        self.shared = inA & inB
        self.only_a = inA & ~inB
        self.only_b = inB & ~inA

        # gather each source's latent onto the union rows it occupies (others left 0)
        featsA = torch.zeros(N, self.C, device=self.device)
        featsB = torch.zeros(N, self.C, device=self.device)
        sa = torch.argsort(keyA)
        featsA[inA] = slat_A.feats[sa[torch.searchsorted(keyA[sa], union[inA])]]
        sb = torch.argsort(keyB)
        featsB[inB] = slat_B.feats[sb[torch.searchsorted(keyB[sb], union[inB])]]
        self.featsA = featsA
        self.featsB = featsB

        # stable per-voxel dissolve thresholds (same across all t -> temporal coherence)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.tau = torch.rand(N, generator=g).to(self.device)

        self.n_a = int(inA.sum())
        self.n_b = int(inB.sum())
        self.n_shared = int(self.shared.sum())
        self.n_union = N

    def at(self, t: float):
        """Return the morphed SLAT (SparseTensor) at fraction ``t`` in ``[0, 1]``."""
        import torch
        from trellis.modules import sparse as sp

        t = float(t)
        if t <= 0.0:
            occ = self.shared | self.only_a
        elif t >= 1.0:
            occ = self.shared | self.only_b
        else:
            occ = self.shared.clone()
            occ = occ | (self.only_a & (self.tau > t))   # A-only stays while t < tau
            occ = occ | (self.only_b & (self.tau < t))   # B-only appears once t > tau

        feats = torch.empty(self.n_union, self.C, device=self.device)
        lerp = (1.0 - t) * self.featsA + t * self.featsB
        feats[self.shared] = lerp[self.shared]
        feats[self.only_a] = self.featsA[self.only_a]
        feats[self.only_b] = self.featsB[self.only_b]

        coords = _coords_from_keys(self.union_keys[occ], self.res)
        return sp.SparseTensor(feats=feats[occ], coords=coords)

    def sequence(self, ts: Sequence[float]):
        """Return ``[self.at(t) for t in ts]``."""
        return [self.at(t) for t in ts]


def morph_sequence(slat_A, slat_B, ts=(0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0,
                   resolution: int = 64):
    """Convenience: build a :class:`SlatMorpher` and return the SLAT for each ``t``."""
    return SlatMorpher(slat_A, slat_B, seed=seed, resolution=resolution).sequence(ts)


def harmonize_slat(pipeline, slat, prompt: str, strength: float = 0.3, seed: int = 0,
                   slat_sampler_params: Optional[dict] = None):
    """Optional SDEdit cleanup: renoise to partial time ``strength`` then denoise.

    Projects an off-manifold latent blend (the shared-voxel lerp) back toward the data
    manifold while staying close to the blend. Reuses the stock slat sampler's
    ``sample_once`` over the tail of its schedule — no submodule edits, no new sampler.

    Args:
        pipeline: a loaded ``TrellisTextTo3DPipeline``.
        slat: a morphed SLAT (denormalized, as :meth:`SlatMorpher.at` returns).
        prompt: text to condition the cleanup denoise (e.g. a blend description).
        strength: starting noise level in ``[0,1]`` — higher = more cleanup, less fidelity.
        seed: RNG seed for the renoise.

    Returns:
        A SLAT (same coords/order) with harmonized latents.
    """
    import torch
    import numpy as np

    device = slat.feats.device
    norm = pipeline.slat_normalization
    mean = torch.tensor(norm["mean"], device=device)[None]
    std = torch.tensor(norm["std"], device=device)[None]
    x0 = slat.replace((slat.feats - mean) / std)            # normalized clean blend

    sampler = pipeline.slat_sampler
    params = {**pipeline.slat_sampler_params, **(slat_sampler_params or {})}
    steps = params.get("steps", 25)
    rescale_t = params.get("rescale_t", 3.0)
    cfg_strength = params.get("cfg_strength", 7.5)
    cfg_interval = params.get("cfg_interval", (0.5, 0.95))

    cond = pipeline.get_cond([prompt])
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    start = int(np.argmin(np.abs(t_seq - strength)))        # step whose t ~ strength
    t_start = float(t_seq[start])

    torch.manual_seed(seed)
    s = sampler.sigma_min + (1 - sampler.sigma_min) * t_start
    sample = x0 * (1 - t_start) + x0.replace(torch.randn_like(x0.feats)) * s

    inject = dict(neg_cond=cond["neg_cond"], cfg_strength=cfg_strength, cfg_interval=cfg_interval)
    model = pipeline.models["slat_flow_model"]
    for i in range(start, steps):
        t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
        sample = sampler.sample_once(model, sample, t, t_prev, cond["cond"], **inject).pred_x_prev

    feats = sample.feats * std + mean
    return slat.replace(feats)
