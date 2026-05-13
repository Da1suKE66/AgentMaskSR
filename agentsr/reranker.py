"""Candidate scoring for observation-constrained token refinement."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .token_masks import TokenMaskSet, token_mask_to_pixel_mask


@dataclass
class CandidateScore:
    candidate_id: int
    seed: Optional[int]
    lr_l1: float
    lr_grad_l1: float
    boundary_l1: float
    edit_penalty_l1: float
    clip_text_similarity: Optional[float]
    clip_image_similarity: Optional[float]
    total: float
    selected: bool = False


@dataclass
class ScoreWeights:
    lr_l1: float = 1.0
    lr_grad_l1: float = 0.25
    boundary_l1: float = 0.50
    edit_penalty_l1: float = 0.50
    clip_text: float = 0.0
    clip_image: float = 0.0


def _rgb_float(image: Image.Image, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _gray(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _grad_mag(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.sqrt(gx * gx + gy * gy)


def lr_l1(candidate: Image.Image, observation: Image.Image) -> float:
    down = candidate.resize(observation.size, Image.Resampling.BICUBIC)
    return float(np.mean(np.abs(_rgb_float(down) - _rgb_float(observation))))


def lr_grad_l1(candidate: Image.Image, observation: Image.Image) -> float:
    down = candidate.resize(observation.size, Image.Resampling.BICUBIC)
    cand_grad = _grad_mag(_gray(_rgb_float(down)))
    obs_grad = _grad_mag(_gray(_rgb_float(observation)))
    return float(np.mean(np.abs(cand_grad - obs_grad)))


def _boundary_ring(mask_image: Image.Image, radius: int = 7) -> np.ndarray:
    gray = mask_image.convert("L")
    size = max(3, 2 * int(radius) + 1)
    dilated = gray.filter(ImageFilter.MaxFilter(size))
    eroded = gray.filter(ImageFilter.MinFilter(size))
    return np.asarray(dilated, dtype=np.int16) != np.asarray(eroded, dtype=np.int16)


def boundary_l1(candidate: Image.Image, previous: Image.Image, active_mask: np.ndarray) -> float:
    mask_image = token_mask_to_pixel_mask(active_mask, candidate.size)
    ring = _boundary_ring(mask_image)
    if not bool(ring.any()):
        return 0.0
    cand = _rgb_float(candidate)
    prev = _rgb_float(previous, candidate.size)
    return float(np.mean(np.abs(cand[ring] - prev[ring])))


def edit_penalty_l1(candidate: Image.Image, previous: Image.Image, protected_mask: np.ndarray) -> float:
    mask_image = token_mask_to_pixel_mask(protected_mask, candidate.size)
    protected = np.asarray(mask_image, dtype=np.uint8) > 0
    if not bool(protected.any()):
        return 0.0
    cand = _rgb_float(candidate)
    prev = _rgb_float(previous, candidate.size)
    return float(np.mean(np.abs(cand[protected] - prev[protected])))


def score_candidate(
    candidate_id: int,
    candidate: Image.Image,
    previous: Image.Image,
    observation: Image.Image,
    masks: TokenMaskSet,
    seed: Optional[int],
    weights: ScoreWeights,
    multimodal_metrics: Optional[Dict[str, Optional[float]]] = None,
) -> CandidateScore:
    lr = lr_l1(candidate, observation)
    grad = lr_grad_l1(candidate, observation)
    boundary = boundary_l1(candidate, previous, masks.active)
    protected = np.logical_or(masks.known, masks.commit)
    edit = edit_penalty_l1(candidate, previous, protected)
    multimodal_metrics = multimodal_metrics or {}
    clip_text = multimodal_metrics.get("clip_text_similarity")
    clip_image = multimodal_metrics.get("clip_image_similarity")
    total = (
        weights.lr_l1 * lr
        + weights.lr_grad_l1 * grad
        + weights.boundary_l1 * boundary
        + weights.edit_penalty_l1 * edit
        - weights.clip_text * float(clip_text if clip_text is not None else 0.0)
        - weights.clip_image * float(clip_image if clip_image is not None else 0.0)
    )
    return CandidateScore(
        candidate_id=candidate_id,
        seed=seed,
        lr_l1=lr,
        lr_grad_l1=grad,
        boundary_l1=boundary,
        edit_penalty_l1=edit,
        clip_text_similarity=clip_text,
        clip_image_similarity=clip_image,
        total=float(total),
    )


def score_candidates(
    candidates: Iterable[Image.Image],
    previous: Image.Image,
    observation: Image.Image,
    masks: TokenMaskSet,
    seeds: Iterable[Optional[int]],
    weights: Optional[ScoreWeights] = None,
    multimodal_metrics: Optional[Iterable[Optional[Dict[str, Optional[float]]]]] = None,
) -> Tuple[int, List[CandidateScore]]:
    weights = weights or ScoreWeights()
    candidate_list = list(candidates)
    seed_list = list(seeds)
    metric_list = list(multimodal_metrics) if multimodal_metrics is not None else [None] * len(candidate_list)
    scores = [
        score_candidate(i, image, previous, observation, masks, seed, weights, metric)
        for i, (image, seed, metric) in enumerate(zip(candidate_list, seed_list, metric_list))
    ]
    if not scores:
        raise ValueError("at least one candidate is required")
    best_id = min(range(len(scores)), key=lambda idx: scores[idx].total)
    scores[best_id].selected = True
    return best_id, scores


def write_scores(path: Path, scores: List[CandidateScore], weights: Optional[ScoreWeights] = None) -> None:
    payload = {
        "weights": asdict(weights or ScoreWeights()),
        "candidates": [asdict(score) for score in scores],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
