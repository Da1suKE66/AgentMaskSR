"""Semantic region priors for active-mask planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


DEFAULT_REFINE_PROMPTS = (
    "fine animal fur texture",
    "hair or fur detail",
    "grass or natural texture",
    "fabric or surface texture",
    "small high frequency visual details",
)

DEFAULT_PROTECT_PROMPTS = (
    "eyes nose mouth face identity",
    "object outline boundary",
    "text logo sign watermark",
    "smooth flat sky wall background",
)


@dataclass
class SemanticPrior:
    refine_score: np.ndarray
    protect_score: np.ndarray
    semantic_score: np.ndarray
    grid_shape: Tuple[int, int]
    refine_prompts: List[str]
    protect_prompts: List[str]
    diagnostics: Dict[str, float | int | List[str]]

    def maps(self) -> Dict[str, np.ndarray]:
        return {
            "semantic_refine": self.refine_score,
            "semantic_protect": self.protect_score,
            "semantic_score": self.semantic_score,
        }


def parse_prompt_list(value: str | Sequence[str] | None, defaults: Sequence[str]) -> List[str]:
    if value is None:
        return list(defaults)
    if isinstance(value, str):
        items = [item.strip() for item in value.replace("|", ";").split(";")]
    else:
        items = [str(item).strip() for item in value]
    return [item for item in items if item]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _grid_crops(image: Image.Image, grid_size: int) -> Tuple[List[Image.Image], List[Tuple[int, int, int, int]]]:
    width, height = image.size
    crops: List[Image.Image] = []
    boxes: List[Tuple[int, int, int, int]] = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            x0 = int(round(gx * width / grid_size))
            y0 = int(round(gy * height / grid_size))
            x1 = int(round((gx + 1) * width / grid_size))
            y1 = int(round((gy + 1) * height / grid_size))
            box = (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))
            crops.append(image.crop(box).convert("RGB"))
            boxes.append(box)
    return crops, boxes


def _upsample_grid(values: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    grid = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255), mode="L")
    return np.asarray(grid.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


class CLIPRegionPrior:
    """Use CLIP crop-text similarities as a coarse semantic mask prior."""

    def __init__(
        self,
        model_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()

    def _text_features(self, prompts: Sequence[str]):
        inputs = self.processor(text=list(prompts), return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _image_features(self, images: Sequence[Image.Image], batch_size: int):
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=list(batch), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            chunks.append(features.detach().cpu())
        return self.torch.cat(chunks, dim=0).to(self.device)

    def build_prior(
        self,
        image: Image.Image,
        refine_prompts: Iterable[str] | None = None,
        protect_prompts: Iterable[str] | None = None,
        grid_size: int = 8,
        batch_size: int = 16,
    ) -> SemanticPrior:
        refine = parse_prompt_list(refine_prompts, DEFAULT_REFINE_PROMPTS)
        protect = parse_prompt_list(protect_prompts, DEFAULT_PROTECT_PROMPTS)
        crops, _boxes = _grid_crops(image.convert("RGB"), int(grid_size))

        image_features = self._image_features(crops, batch_size=max(1, int(batch_size)))
        refine_features = self._text_features(refine)
        protect_features = self._text_features(protect)

        refine_sim = (image_features @ refine_features.T).max(dim=1).values.detach().cpu().numpy()
        protect_sim = (image_features @ protect_features.T).max(dim=1).values.detach().cpu().numpy()

        refine_grid = _normalize(refine_sim.reshape(grid_size, grid_size))
        protect_grid = _normalize(protect_sim.reshape(grid_size, grid_size))
        semantic_grid = _normalize(refine_grid - 0.75 * protect_grid)

        refine_map = _upsample_grid(refine_grid, image.size)
        protect_map = _upsample_grid(protect_grid, image.size)
        semantic_map = _upsample_grid(semantic_grid, image.size)
        diagnostics: Dict[str, float | int | List[str]] = {
            "semantic_grid_size": int(grid_size),
            "semantic_refine_mean": float(refine_map.mean()),
            "semantic_protect_mean": float(protect_map.mean()),
            "semantic_score_mean": float(semantic_map.mean()),
            "semantic_refine_prompts": refine,
            "semantic_protect_prompts": protect,
        }
        return SemanticPrior(
            refine_score=refine_map,
            protect_score=protect_map,
            semantic_score=semantic_map,
            grid_shape=(grid_size, grid_size),
            refine_prompts=refine,
            protect_prompts=protect,
            diagnostics=diagnostics,
        )
