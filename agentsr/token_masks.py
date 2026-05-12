"""Token-grid mask utilities for Meissonic-SR.

The controller may start from pixel-space evidence such as edge/detail maps,
but the sampler contract is token-space: known tokens are never masked, active
tokens may be resampled, and committed tokens are frozen after they pass the
round-level consistency checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


DEFAULT_VAE_SCALE_FACTOR = 16


@dataclass
class TokenMaskSet:
    """Boolean token masks used by the token-rerank SR loop."""

    known: np.ndarray
    active: np.ndarray
    commit: np.ndarray
    outpaint: np.ndarray

    def __post_init__(self) -> None:
        shape = self.active.shape
        for name, value in {
            "known": self.known,
            "commit": self.commit,
            "outpaint": self.outpaint,
        }.items():
            if value.shape != shape:
                raise ValueError(f"{name} mask shape {value.shape} does not match active mask shape {shape}")

        self.known = self.known.astype(bool)
        self.active = self.active.astype(bool)
        self.commit = self.commit.astype(bool)
        self.outpaint = self.outpaint.astype(bool)
        self.active = np.logical_and(self.active, ~np.logical_or(self.known, self.commit))

    @property
    def shape(self) -> Tuple[int, int]:
        return self.active.shape

    @property
    def active_ratio(self) -> float:
        return float(self.active.mean())

    @property
    def commit_ratio(self) -> float:
        return float(self.commit.mean())

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "known_mask": self.known,
            "active_mask": self.active,
            "commit_mask": self.commit,
            "outpaint_mask": self.outpaint,
        }


def infer_token_shape(target_size: Sequence[int], vae_scale_factor: int = DEFAULT_VAE_SCALE_FACTOR) -> Tuple[int, int]:
    """Return token grid shape as (height, width) for a target image size."""

    width, height = int(target_size[0]), int(target_size[1])
    if width % vae_scale_factor != 0 or height % vae_scale_factor != 0:
        raise ValueError(
            f"target size {(width, height)} must be divisible by vae scale factor {vae_scale_factor}"
        )
    return (height // vae_scale_factor, width // vae_scale_factor)


def pixel_mask_to_token_mask(mask_image: Image.Image, token_shape: Tuple[int, int], threshold: float = 0.5) -> np.ndarray:
    """Downsample a white-repaint pixel mask to token space."""

    token_h, token_w = token_shape
    resized = mask_image.convert("L").resize((token_w, token_h), Image.Resampling.BOX)
    return (np.asarray(resized, dtype=np.float32) / 255.0) >= float(threshold)


def token_mask_to_pixel_mask(token_mask: np.ndarray, target_size: Tuple[int, int]) -> Image.Image:
    """Convert a token mask to a Meissonic-compatible white-repaint pixel mask."""

    mask = Image.fromarray(np.uint8(token_mask.astype(bool)) * 255, mode="L")
    return mask.resize(target_size, Image.Resampling.NEAREST)


def build_initial_token_masks(
    mask_image: Image.Image,
    target_size: Tuple[int, int],
    vae_scale_factor: int = DEFAULT_VAE_SCALE_FACTOR,
    outpaint_mask: Optional[np.ndarray] = None,
) -> TokenMaskSet:
    """Create initial known/active/commit/outpaint masks from controller output."""

    token_shape = infer_token_shape(target_size, vae_scale_factor=vae_scale_factor)
    active = pixel_mask_to_token_mask(mask_image, token_shape)
    outpaint = np.zeros(token_shape, dtype=bool)
    if outpaint_mask is not None:
        outpaint_img = Image.fromarray(np.uint8(outpaint_mask.astype(bool)) * 255, mode="L")
        outpaint = pixel_mask_to_token_mask(outpaint_img, token_shape)
        active = np.logical_or(active, outpaint)

    commit = np.zeros(token_shape, dtype=bool)
    known = ~active
    return TokenMaskSet(known=known, active=active, commit=commit, outpaint=outpaint)


def _rgb_float(image: Image.Image, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _local_lr_worse_tokens(
    previous: Image.Image,
    candidate: Image.Image,
    observation: Image.Image,
    token_shape: Tuple[int, int],
    margin: float,
) -> np.ndarray:
    prev_down = previous.resize(observation.size, Image.Resampling.BICUBIC)
    cand_down = candidate.resize(observation.size, Image.Resampling.BICUBIC)
    obs = _rgb_float(observation)
    prev_err = np.mean(np.abs(_rgb_float(prev_down) - obs), axis=2)
    cand_err = np.mean(np.abs(_rgb_float(cand_down) - obs), axis=2)
    worse = cand_err > (prev_err + float(margin))
    worse_img = Image.fromarray(np.uint8(worse) * 255, mode="L")
    return pixel_mask_to_token_mask(worse_img, token_shape, threshold=0.25)


def candidate_agreement_mask(candidate_tokens: Iterable[np.ndarray], best_tokens: np.ndarray) -> np.ndarray:
    """Return tokens where all candidates agree with the selected candidate."""

    best = np.asarray(best_tokens)
    best_view = best[0] if best.ndim == 3 else best
    agreement = np.ones(best_view.shape[-2:], dtype=bool)
    for tokens in candidate_tokens:
        arr = np.asarray(tokens)
        arr_view = arr[0] if arr.ndim == 3 else arr
        agreement = np.logical_and(agreement, arr_view == best_view)
    return agreement


def update_masks_after_round(
    masks: TokenMaskSet,
    previous_image: Image.Image,
    best_image: Image.Image,
    observation: Image.Image,
    candidate_tokens: Iterable[np.ndarray],
    best_tokens: np.ndarray,
    lr_worse_margin: float = 1.0,
) -> Tuple[TokenMaskSet, Dict[str, float]]:
    """Commit stable active tokens and keep/remask unstable active tokens."""

    agreement = candidate_agreement_mask(candidate_tokens, best_tokens)
    worse = _local_lr_worse_tokens(previous_image, best_image, observation, masks.shape, margin=lr_worse_margin)

    stable = np.logical_and.reduce((masks.active, agreement, ~worse))
    new_commit = np.logical_or(masks.commit, stable)
    new_active = np.logical_and(masks.active, ~stable)
    new_active = np.logical_or(new_active, np.logical_and(worse, ~masks.known))
    new_active = np.logical_and(new_active, ~np.logical_or(masks.known, new_commit))

    updated = TokenMaskSet(
        known=masks.known,
        active=new_active,
        commit=new_commit,
        outpaint=masks.outpaint,
    )
    diagnostics = {
        "agreement_ratio": float(agreement.mean()),
        "lr_worse_ratio": float(worse.mean()),
        "newly_committed_ratio": float(stable.mean()),
        "active_ratio": updated.active_ratio,
        "commit_ratio": updated.commit_ratio,
    }
    return updated, diagnostics


def save_token_masks_npz(
    path: Path,
    masks: TokenMaskSet,
    z_base: Optional[np.ndarray] = None,
    z_current: Optional[np.ndarray] = None,
) -> None:
    """Persist token masks and optional token ids for reproducibility."""

    payload = masks.to_arrays()
    if z_base is not None:
        payload["z_base"] = np.asarray(z_base)
    if z_current is not None:
        payload["z_current"] = np.asarray(z_current)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def mask_metadata(masks: TokenMaskSet) -> Dict[str, object]:
    return {
        "token_shape": list(masks.shape),
        "known_ratio": float(masks.known.mean()),
        "active_ratio": masks.active_ratio,
        "commit_ratio": masks.commit_ratio,
        "outpaint_ratio": float(masks.outpaint.mean()),
        "active_tokens": int(masks.active.sum()),
        "commit_tokens": int(masks.commit.sum()),
    }
