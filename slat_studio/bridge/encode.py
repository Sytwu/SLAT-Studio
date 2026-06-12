"""DINOv2 feature extraction + SLAT VAE encoding for the encoding bridge.

This mirrors TRELLIS's ``dataset_toolkits/extract_feature.py`` and ``encode_latent.py``,
but runs fully **in memory** (no metadata.csv / per-asset file dance) so it can be composed
into a live pipeline. It reuses only public TRELLIS / utils3d APIs — nothing under the
submodule is modified.

Pipeline: multiview RGB renders + their cameras + occupied voxel indices
  -> DINOv2 ``dinov2_vitl14_reg`` patch tokens per view
  -> project each voxel center into every view, bilinearly sample its patch token
  -> mean over views  => per-voxel [1024] feature
  -> frozen SLAT VAE encoder => per-voxel [8] latent (a SLAT SparseTensor).
"""
import numpy as np
import torch
import torch.nn.functional as F
import utils3d
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DINO_RES = 518
_PATCH = 14
_N_PATCH = _DINO_RES // _PATCH  # 37
_DINO_DIM = 1024  # vitl14 token dim

# SLAT VAE encoder shipped with TRELLIS (same VAE used by image- and text- models).
DEFAULT_ENCODER = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
DEFAULT_DINOV2 = "dinov2_vitl14_reg"


def load_dinov2(model_name=DEFAULT_DINOV2, device="cuda"):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    return model.eval().to(device)


def load_slat_encoder(pretrained=DEFAULT_ENCODER, device="cuda"):
    import trellis.models as models
    return models.from_pretrained(pretrained).eval().to(device)


def _prep_images(images, device):
    """[V,H,W,3] (uint8 or float) renders on a black bg -> normalized [V,3,518,518]."""
    norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    out = []
    for im in images:
        if isinstance(im, np.ndarray):
            im = torch.from_numpy(np.ascontiguousarray(im))
        im = im.float()
        if im.max() > 1.5:  # uint8 range -> [0,1]
            im = im / 255.0
        im = im[..., :3].permute(2, 0, 1)  # [3,H,W]; black bg == premultiplied
        im = F.interpolate(im[None], size=(_DINO_RES, _DINO_RES),
                           mode="bilinear", align_corners=False)[0]
        out.append(norm(im))
    return torch.stack(out).to(device)


def _stack_cams(cams, device):
    if isinstance(cams, (list, tuple)):
        cams = torch.stack([c if torch.is_tensor(c) else torch.tensor(c) for c in cams])
    return cams.to(device).float()


@torch.no_grad()
def extract_voxel_features(images, extrinsics, intrinsics, coords,
                           dinov2, resolution=64, batch_size=16, device="cuda"):
    """Per-voxel DINOv2 feature, averaged over all views (mirrors extract_feature.py).

    Args:
        images: ``[V,H,W,3]`` renders (black background) matching the cameras.
        extrinsics/intrinsics: per-view CV cameras (lists or stacked tensors) — the SAME
            ones used to produce ``images`` (e.g. from ``render_utils.render_multiview``).
        coords: ``[N,3]`` occupied voxel indices in ``[0, resolution)``.
        dinov2: a loaded ``dinov2_vitl14_reg`` model.

    Returns:
        ``[N, 1024]`` float32 features on ``device``.
    """
    extrinsics = _stack_cams(extrinsics, device)
    intrinsics = _stack_cams(intrinsics, device)
    imgs = _prep_images(images, device)
    V = imgs.shape[0]

    coords = coords.to(device).long()
    positions = (coords.float() + 0.5) / resolution - 0.5  # [N,3] in [-0.5,0.5]
    N = positions.shape[0]

    feat_sum = torch.zeros(N, _DINO_DIM, device=device, dtype=torch.float32)
    for i in range(0, V, batch_size):
        bimg = imgs[i:i + batch_size]
        bext = extrinsics[i:i + batch_size]
        bint = intrinsics[i:i + batch_size]
        bs = bimg.shape[0]

        features = dinov2(bimg, is_training=True)
        patchtokens = features["x_prenorm"][:, dinov2.num_register_tokens + 1:]
        patchtokens = patchtokens.permute(0, 2, 1).reshape(bs, _DINO_DIM, _N_PATCH, _N_PATCH)

        uv = utils3d.torch.project_cv(positions, bext, bint)[0] * 2 - 1  # [bs,N,2] in [-1,1]
        sampled = F.grid_sample(patchtokens, uv.unsqueeze(1),
                                mode="bilinear", align_corners=False)  # [bs,1024,1,N]
        sampled = sampled.squeeze(2)  # [bs,1024,N]
        feat_sum += sampled.sum(dim=0).permute(1, 0).float()  # [N,1024]

    return feat_sum / V


@torch.no_grad()
def encode_to_slat(features, coords, encoder, device="cuda"):
    """Frozen SLAT VAE encode (mirrors encode_latent.py) -> SLAT SparseTensor [N,8]."""
    import trellis.modules.sparse as sp
    N = features.shape[0]
    coords4 = torch.cat([
        torch.zeros(N, 1, dtype=torch.int32, device=device),
        coords.to(device).int(),
    ], dim=1)
    feats_in = sp.SparseTensor(feats=features.float().to(device), coords=coords4)
    latent = encoder(feats_in, sample_posterior=False)
    assert torch.isfinite(latent.feats).all(), "Non-finite latent from encoder"
    return latent
