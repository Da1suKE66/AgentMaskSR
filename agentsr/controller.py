"""Training-free controller for Meissonic image refinement.

The controller turns a low-resolution observation and a user instruction into:

- a structured agent plan;
- a target-size initialization image;
- a Meissonic-compatible binary mask image;
- lightweight observation-consistency diagnostics.

It intentionally does not train or modify Meissonic. The current backend uses
deterministic image statistics as the local agent fallback. A VLM/API planner
can later fill the same :class:`AgentPlan` schema without changing the sampler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, low res, blurry, distortion, watermark, logo, "
    "signature, text artifacts, jpeg artifacts, duplicate, ugly"
)


@dataclass
class AgentPlan:
    """Structured control plan consumed by the training-free controller."""

    mode: str = "sr"
    alpha: float = 0.45
    target_resolution: Tuple[int, int] = (1024, 1024)
    protected_regions: List[str] = field(default_factory=lambda: ["main object", "global structure"])
    enhance_regions: List[str] = field(default_factory=lambda: ["edges", "fine texture", "high-frequency detail"])
    outpaint_direction: List[str] = field(default_factory=list)
    lr_consistency_weight: float = 0.85
    boundary_consistency_weight: float = 0.70
    temperature: float = 0.45
    mask_policy: str = "frequency_entropy_attention"
    prompt: str = (
        "restore faithful high-frequency details while preserving the low-resolution observation"
    )
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["target_resolution"] = list(self.target_resolution)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AgentPlan":
        payload = dict(data)
        if "target_resolution" in payload:
            payload["target_resolution"] = tuple(payload["target_resolution"])
        return cls(**payload)


def parse_resolution(value: str | Sequence[int] | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, str):
        clean = value.lower().replace(" ", "")
        if "x" not in clean:
            side = int(clean)
            return (side, side)
        width, height = clean.split("x", 1)
        return (int(width), int(height))
    if len(value) != 2:
        raise ValueError("target resolution must contain width and height")
    return (int(value[0]), int(value[1]))


def derive_agent_plan(
    instruction: str,
    target_resolution: str | Sequence[int] = "1024x1024",
    mode: Optional[str] = None,
    alpha: Optional[float] = None,
    outpaint_direction: Optional[Iterable[str]] = None,
) -> AgentPlan:
    """Create a deterministic plan from user intent.

    This is the open, reproducible fallback for the agent layer. It keeps the
    model-generation contribution inside Meissonic and the controller instead
    of delegating final image synthesis to a closed API.
    """

    text = instruction.lower()
    if mode is not None:
        inferred_mode = mode
    elif any(word in text for word in ("outpaint", "expand", "extend", "扩图", "外扩")):
        inferred_mode = "sr_outpaint" if any(word in text for word in ("sr", "super", "超分")) else "outpaint"
    elif any(word in text for word in ("detail", "texture", "细节", "增强")):
        inferred_mode = "detail"
    else:
        inferred_mode = "sr"

    if alpha is None:
        if inferred_mode == "sr":
            alpha = 0.35
        elif inferred_mode == "detail":
            alpha = 0.55
        elif inferred_mode == "outpaint":
            alpha = 0.70
        else:
            alpha = 0.62

    directions = list(outpaint_direction or [])
    if not directions and inferred_mode in {"outpaint", "sr_outpaint"}:
        for key in ("left", "right", "top", "bottom"):
            if key in text:
                directions.append(key)
        if not directions:
            directions = ["left", "right", "top", "bottom"]

    alpha = float(np.clip(alpha, 0.0, 1.0))
    consistency = float(np.interp(alpha, [0.0, 1.0], [0.95, 0.65]))
    boundary = float(np.interp(alpha, [0.0, 1.0], [0.85, 0.55]))
    temperature = float(np.interp(alpha, [0.0, 1.0], [0.25, 0.75]))

    return AgentPlan(
        mode=inferred_mode,
        alpha=alpha,
        target_resolution=parse_resolution(target_resolution),
        outpaint_direction=directions,
        lr_consistency_weight=consistency,
        boundary_consistency_weight=boundary,
        temperature=temperature,
        prompt=instruction.strip() or AgentPlan().prompt,
    )


def load_plan(path: Path) -> AgentPlan:
    with path.open("r", encoding="utf-8") as handle:
        return AgentPlan.from_mapping(json.load(handle))


def save_plan(plan: AgentPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    img = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), mode="L")
    return np.asarray(img.filter(ImageFilter.BoxBlur(radius)), dtype=np.float32) / 255.0


def frequency_entropy_map(image: Image.Image) -> np.ndarray:
    """Compute a normalized detail map from gradients and local variance."""

    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    gradient = np.sqrt(gx * gx + gy * gy)

    local_mean = _box_blur(gray, radius=4)
    local_sq_mean = _box_blur(gray * gray, radius=4)
    variance = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)

    detail = 0.65 * _normalize_array(gradient) + 0.35 * _normalize_array(variance)
    return _normalize_array(detail)


def make_outpaint_canvas(
    image: Image.Image,
    target_size: Tuple[int, int],
    outpaint_direction: Sequence[str],
    margin_ratio: float,
) -> Tuple[Image.Image, np.ndarray, Tuple[int, int, int, int]]:
    """Place an upscaled observation on a larger target canvas."""

    width, height = target_size
    directions = set(outpaint_direction)
    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))

    left = margin_x if "left" in directions else 0
    right = margin_x if "right" in directions else 0
    top = margin_y if "top" in directions else 0
    bottom = margin_y if "bottom" in directions else 0

    content_w = max(16, width - left - right)
    content_h = max(16, height - top - bottom)

    resized = image.resize((content_w, content_h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", target_size, tuple(np.asarray(resized).reshape(-1, 3).mean(axis=0).astype(np.uint8)))
    canvas.paste(resized, (left, top))

    outpaint_mask = np.ones((height, width), dtype=bool)
    outpaint_mask[top : top + content_h, left : left + content_w] = False
    bbox = (left, top, left + content_w, top + content_h)
    return canvas, outpaint_mask, bbox


def adaptive_mask(
    init_image: Image.Image,
    plan: AgentPlan,
    outpaint_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    detail = frequency_entropy_map(init_image)
    alpha = float(np.clip(plan.alpha, 0.0, 1.0))

    if plan.mode == "sr":
        budget = 0.10 + 0.30 * alpha
    elif plan.mode == "detail":
        budget = 0.20 + 0.45 * alpha
    elif plan.mode == "outpaint":
        budget = 0.04 + 0.16 * alpha
    else:
        budget = 0.14 + 0.36 * alpha

    threshold = float(np.quantile(detail, max(0.0, 1.0 - budget)))
    mask = detail >= threshold

    if outpaint_mask is not None:
        mask = np.logical_or(mask, outpaint_mask)
        feather = Image.fromarray(np.uint8(outpaint_mask) * 255, mode="L").filter(ImageFilter.GaussianBlur(6))
        feather_arr = np.asarray(feather, dtype=np.float32) / 255.0
        boundary_band = feather_arr > 0.05
        if plan.boundary_consistency_weight >= 0.70:
            mask = np.logical_or(mask, np.logical_and(boundary_band, detail > np.quantile(detail, 0.65)))

    mask_img = Image.fromarray(np.uint8(mask) * 255, mode="L")
    return mask_img.filter(ImageFilter.MaxFilter(3))


def downsample_consistency_metrics(candidate: Image.Image, observation: Image.Image) -> Dict[str, float]:
    down = candidate.resize(observation.size, Image.Resampling.BICUBIC)
    a = np.asarray(down, dtype=np.float32)
    b = np.asarray(observation.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return {"mse_downsample_vs_lr": mse, "psnr_downsample_vs_lr": psnr}


def tile_grid(
    target_size: Tuple[int, int],
    tile_size: int = 1024,
    overlap: int = 128,
) -> List[Dict[str, int]]:
    width, height = target_size
    stride = max(1, tile_size - overlap)
    tiles: List[Dict[str, int]] = []
    for y in range(0, max(1, height - overlap), stride):
        for x in range(0, max(1, width - overlap), stride):
            x0 = min(x, max(0, width - tile_size))
            y0 = min(y, max(0, height - tile_size))
            x1 = min(width, x0 + tile_size)
            y1 = min(height, y0 + tile_size)
            tile = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            if tile not in tiles:
                tiles.append(tile)
    return tiles


def build_refinement_assets(
    input_image: Image.Image,
    plan: AgentPlan,
    output_dir: Path,
    outpaint_margin_ratio: float = 0.18,
    tile_size: int = 1024,
    tile_overlap: int = 128,
) -> Dict[str, Any]:
    """Build and save controller assets for Meissonic refinement."""

    output_dir.mkdir(parents=True, exist_ok=True)
    observation = input_image.convert("RGB")
    target_size = plan.target_resolution

    outpaint_mask = None
    protected_bbox = (0, 0, target_size[0], target_size[1])
    if plan.mode in {"outpaint", "sr_outpaint"}:
        init_image, outpaint_mask, protected_bbox = make_outpaint_canvas(
            observation,
            target_size,
            plan.outpaint_direction,
            outpaint_margin_ratio,
        )
    else:
        init_image = observation.resize(target_size, Image.Resampling.BICUBIC)

    mask_image = adaptive_mask(init_image, plan, outpaint_mask=outpaint_mask)
    metrics = downsample_consistency_metrics(init_image, observation)

    paths = {
        "init_image": output_dir / "init_observation.png",
        "mask_image": output_dir / "mask_refine.png",
        "plan": output_dir / "agent_plan.json",
        "metrics": output_dir / "controller_metrics.json",
    }

    init_image.save(paths["init_image"])
    mask_image.save(paths["mask_image"])
    save_plan(plan, paths["plan"])

    mask_arr = np.asarray(mask_image, dtype=np.uint8) > 0
    diagnostics: Dict[str, Any] = {
        **metrics,
        "mode": plan.mode,
        "alpha": plan.alpha,
        "target_resolution": list(target_size),
        "masked_pixel_ratio": float(mask_arr.mean()),
        "protected_bbox": list(protected_bbox),
        "tile_grid": tile_grid(target_size, tile_size=tile_size, overlap=tile_overlap),
    }
    with paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return {
        "init_image": init_image,
        "mask_image": mask_image,
        "plan": plan,
        "diagnostics": diagnostics,
        "paths": {key: str(value) for key, value in paths.items()},
    }
