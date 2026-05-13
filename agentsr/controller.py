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

MASK_STRATEGIES = (
    "frequency",
    "edge",
    "variance",
    "hybrid",
    "uncage",
    "semantic_uncage",
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
    mask_policy: str = "uncage"
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


def detail_evidence_maps(image: Image.Image) -> Dict[str, np.ndarray]:
    """Compute deterministic evidence maps used by mask strategies."""

    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    gradient = _normalize_array(np.sqrt(gx * gx + gy * gy))

    local_mean = _box_blur(gray, radius=4)
    local_sq_mean = _box_blur(gray * gray, radius=4)
    variance = _normalize_array(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

    frequency = _normalize_array(0.65 * gradient + 0.35 * variance)
    texture = _normalize_array(0.25 * gradient + 0.75 * variance)
    flatness = _normalize_array(1.0 - frequency)
    return {
        "gray": gray,
        "gradient": gradient,
        "variance": variance,
        "frequency": frequency,
        "texture": texture,
        "flatness": flatness,
    }


def frequency_entropy_map(image: Image.Image) -> np.ndarray:
    """Compute a normalized detail map from gradients and local variance."""

    return detail_evidence_maps(image)["frequency"]


def mask_budget_for_plan(plan: AgentPlan) -> float:
    """Return the target pixel mask budget before token-grid quantization."""

    alpha = float(np.clip(plan.alpha, 0.0, 1.0))
    if plan.mode == "sr":
        return 0.06 + 0.18 * alpha
    if plan.mode == "detail":
        return 0.16 + 0.36 * alpha
    if plan.mode == "outpaint":
        return 0.04 + 0.16 * alpha
    return 0.14 + 0.36 * alpha


def _deterministic_blue_noise(shape: Tuple[int, int]) -> np.ndarray:
    """Cheap deterministic hash field used for sparse uncage sampling."""

    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    hashed = np.sin((xx + 0.5) * 12.9898 + (yy + 0.5) * 78.233) * 43758.5453
    return (hashed - np.floor(hashed)).astype(np.float32)


def mask_score_map(
    init_image: Image.Image,
    strategy: str,
    semantic_prior: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return a normalized score map for a named mask strategy."""

    strategy = normalize_mask_strategy(strategy)
    maps = detail_evidence_maps(init_image)
    gradient = maps["gradient"]
    variance = maps["variance"]
    frequency = maps["frequency"]
    blue = _deterministic_blue_noise(frequency.shape)

    if strategy == "frequency":
        score = frequency
    elif strategy == "edge":
        score = gradient
    elif strategy == "variance":
        score = variance
    elif strategy == "hybrid":
        score = 0.50 * frequency + 0.30 * variance + 0.15 * gradient + 0.05 * blue
    elif strategy in {"uncage", "semantic_uncage"}:
        # Uncage keeps the strongest structure lines as a cage while releasing
        # sparse texture/detail islands around them. This avoids large connected
        # redraw areas in conservative SR.
        strong_edge = gradient >= np.quantile(gradient, 0.92)
        base_score = 0.45 * variance + 0.35 * frequency + 0.20 * blue
        base_score = np.where(strong_edge & (variance < np.quantile(variance, 0.70)), base_score * 0.35, base_score)
        if strategy == "semantic_uncage" and semantic_prior is not None:
            semantic_score = np.asarray(semantic_prior.get("semantic_score"), dtype=np.float32)
            protect_score = np.asarray(semantic_prior.get("semantic_protect"), dtype=np.float32)
            if semantic_score.shape != base_score.shape:
                raise ValueError(
                    f"semantic score shape {semantic_score.shape} does not match mask score shape {base_score.shape}"
                )
            maps["semantic_score"] = _normalize_array(semantic_score)
            maps["semantic_protect"] = _normalize_array(protect_score)
            base_norm = _normalize_array(base_score)
            score = base_norm * (0.85 + 0.45 * maps["semantic_score"])
            score = score - 0.40 * maps["semantic_protect"]
            high_protect = maps["semantic_protect"] >= np.quantile(maps["semantic_protect"], 0.75)
            score = np.where(high_protect, score * 0.55, score)
        else:
            score = base_score
    else:
        raise ValueError(f"unknown mask strategy: {strategy}")
    return _normalize_array(score), maps


def normalize_mask_strategy(strategy: Optional[str]) -> str:
    if not strategy:
        return "uncage"
    clean = strategy.lower().strip().replace("-", "_")
    aliases = {
        "frequency_entropy_attention": "frequency",
        "freq": "frequency",
        "edges": "edge",
        "edge_only": "edge",
        "texture": "variance",
        "local_variance": "variance",
        "mixed": "hybrid",
        "blue_noise": "uncage",
        "uncaged": "uncage",
        "semantic": "semantic_uncage",
        "vlm_uncage": "semantic_uncage",
        "clip_uncage": "semantic_uncage",
    }
    clean = aliases.get(clean, clean)
    if clean not in MASK_STRATEGIES:
        raise ValueError(f"mask strategy must be one of {', '.join(MASK_STRATEGIES)}, got {strategy!r}")
    return clean


def _threshold_by_budget(score: np.ndarray, budget: float) -> np.ndarray:
    budget = float(np.clip(budget, 0.0, 1.0))
    if budget <= 0.0:
        return np.zeros(score.shape, dtype=bool)
    if budget >= 1.0:
        return np.ones(score.shape, dtype=bool)
    threshold = float(np.quantile(score, max(0.0, 1.0 - budget)))
    return score >= threshold


def mask_quality_diagnostics(
    mask: np.ndarray,
    score: np.ndarray,
    maps: Mapping[str, np.ndarray],
    budget: float,
    strategy: str,
) -> Dict[str, float | str]:
    active = mask.astype(bool)
    if not bool(active.any()):
        return {
            "mask_strategy": strategy,
            "mask_budget": float(budget),
            "mask_score_mean_active": 0.0,
            "mask_score_mean_frozen": float(score.mean()),
            "detail_coverage_top25": 0.0,
            "edge_coverage_top10": 0.0,
            "texture_coverage_top25": 0.0,
            "flat_leakage_bottom25": 0.0,
        }

    frequency = maps["frequency"]
    gradient = maps["gradient"]
    variance = maps["variance"]
    flatness = maps["flatness"]
    high_detail = frequency >= np.quantile(frequency, 0.75)
    strong_edge = gradient >= np.quantile(gradient, 0.90)
    high_texture = variance >= np.quantile(variance, 0.75)
    flat = flatness >= np.quantile(flatness, 0.75)
    frozen = ~active
    return {
        "mask_strategy": strategy,
        "mask_budget": float(budget),
        "mask_score_mean_active": float(score[active].mean()),
        "mask_score_mean_frozen": float(score[frozen].mean()) if bool(frozen.any()) else 0.0,
        "detail_coverage_top25": float(np.logical_and(active, high_detail).sum() / max(1, int(high_detail.sum()))),
        "edge_coverage_top10": float(np.logical_and(active, strong_edge).sum() / max(1, int(strong_edge.sum()))),
        "texture_coverage_top25": float(np.logical_and(active, high_texture).sum() / max(1, int(high_texture.sum()))),
        "flat_leakage_bottom25": float(np.logical_and(active, flat).sum() / max(1, int(active.sum()))),
        **(
            {
                "semantic_score_mean_active": float(maps["semantic_score"][active].mean()),
                "semantic_protect_mean_active": float(maps["semantic_protect"][active].mean()),
                "semantic_protect_top25_leakage": float(
                    np.logical_and(
                        active,
                        maps["semantic_protect"] >= np.quantile(maps["semantic_protect"], 0.75),
                    ).sum()
                    / max(1, int(active.sum()))
                ),
            }
            if "semantic_score" in maps and "semantic_protect" in maps
            else {}
        ),
    }


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
    strategy: Optional[str] = None,
    semantic_prior: Optional[Mapping[str, np.ndarray]] = None,
) -> Image.Image:
    strategy = normalize_mask_strategy(strategy or plan.mask_policy)
    score, maps = mask_score_map(init_image, strategy, semantic_prior=semantic_prior)
    budget = mask_budget_for_plan(plan)
    mask = _threshold_by_budget(score, budget)

    if outpaint_mask is not None:
        mask = np.logical_or(mask, outpaint_mask)
        feather = Image.fromarray(np.uint8(outpaint_mask) * 255, mode="L").filter(ImageFilter.GaussianBlur(6))
        feather_arr = np.asarray(feather, dtype=np.float32) / 255.0
        boundary_band = feather_arr > 0.05
        if plan.boundary_consistency_weight >= 0.70:
            detail = maps["frequency"]
            mask = np.logical_or(mask, np.logical_and(boundary_band, detail > np.quantile(detail, 0.65)))

    mask_img = Image.fromarray(np.uint8(mask) * 255, mode="L")
    if plan.mode in {"outpaint", "sr_outpaint"}:
        return mask_img.filter(ImageFilter.MaxFilter(3))
    return mask_img


def save_mask_strategy_artifacts(
    init_image: Image.Image,
    mask_image: Image.Image,
    output_dir: Path,
    strategy: str,
    semantic_prior: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, str]:
    score, maps = mask_score_map(init_image, strategy, semantic_prior=semantic_prior)
    score_img = Image.fromarray(np.uint8(score * 255), mode="L")
    frequency_img = Image.fromarray(np.uint8(maps["frequency"] * 255), mode="L")
    overlay = init_image.convert("RGBA")
    mask_arr = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    red = Image.new("RGBA", init_image.size, (255, 40, 40, 92))
    transparent = Image.new("RGBA", init_image.size, (0, 0, 0, 0))
    mask_overlay = Image.composite(red, transparent, Image.fromarray(np.uint8(mask_arr) * 255, mode="L"))
    overlay = Image.alpha_composite(overlay, mask_overlay)

    paths = {
        "mask_score": output_dir / "mask_score.png",
        "frequency_map": output_dir / "frequency_map.png",
        "mask_overlay": output_dir / "mask_overlay.png",
    }
    score_img.save(paths["mask_score"])
    frequency_img.save(paths["frequency_map"])
    overlay.convert("RGB").save(paths["mask_overlay"])
    if "semantic_score" in maps:
        semantic_score_path = output_dir / "semantic_score_map.png"
        semantic_protect_path = output_dir / "semantic_protect_map.png"
        Image.fromarray(np.uint8(maps["semantic_score"] * 255), mode="L").save(semantic_score_path)
        Image.fromarray(np.uint8(maps["semantic_protect"] * 255), mode="L").save(semantic_protect_path)
        paths["semantic_score_map"] = semantic_score_path
        paths["semantic_protect_map"] = semantic_protect_path
    return {key: str(value) for key, value in paths.items()}


def downsample_consistency_metrics(candidate: Image.Image, observation: Image.Image) -> Dict[str, float]:
    down = candidate.resize(observation.size, Image.Resampling.BICUBIC)
    a = np.asarray(down, dtype=np.float32)
    b = np.asarray(observation.convert("RGB"), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse <= 1e-12 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return {"mse_downsample_vs_lr": mse, "psnr_downsample_vs_lr": psnr}


def _rgb_float(image: Image.Image, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _float_rgb_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255).round()), mode="RGB")


def _soft_mask(mask_image: Image.Image, size: Tuple[int, int], blur_radius: float) -> np.ndarray:
    mask = mask_image.convert("L").resize(size, Image.Resampling.BICUBIC)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return np.clip(arr[..., None], 0.0, 1.0)


def observation_consistency_project(
    candidate: Image.Image,
    observation: Image.Image,
    init_image: Image.Image,
    mask_image: Optional[Image.Image] = None,
    lr_weight: float = 0.85,
    edit_strength: float = 0.60,
    num_steps: int = 3,
    mask_blur_radius: float = 6.0,
) -> Tuple[Image.Image, Dict[str, float]]:
    """Project an edited image back toward the LR observation.

    This is a pixel-space proxy for observation-constrained token refinement:

    1. Downsample the current HR candidate to the LR observation size.
    2. Compute the low-frequency residual against the LR observation.
    3. Upsample that residual and subtract it from the HR candidate.
    4. Re-apply the controller mask so unmasked/protected regions stay close to
       the bicubic observation initialization.

    The function is intentionally deterministic and training-free.
    """

    target_size = init_image.size
    observation = observation.convert("RGB")
    current = _rgb_float(candidate, target_size)
    init_arr = _rgb_float(init_image, target_size)
    obs_arr = _rgb_float(observation)

    before = downsample_consistency_metrics(_float_rgb_image(current), observation)
    lr_weight = float(np.clip(lr_weight, 0.0, 1.0))
    edit_strength = float(np.clip(edit_strength, 0.0, 1.0))

    for _ in range(max(0, int(num_steps))):
        down = _float_rgb_image(current).resize(observation.size, Image.Resampling.BICUBIC)
        residual = _rgb_float(down) - obs_arr
        residual_up = _rgb_float(_float_rgb_image(residual + 127.5).resize(target_size, Image.Resampling.BICUBIC)) - 127.5
        current = np.clip(current - lr_weight * residual_up, 0, 255)

    if mask_image is not None:
        allowed = _soft_mask(mask_image, target_size, mask_blur_radius) * edit_strength
        current = init_arr * (1.0 - allowed) + current * allowed

    projected = _float_rgb_image(current)
    after = downsample_consistency_metrics(projected, observation)
    diagnostics = {
        "projection_steps": int(num_steps),
        "projection_lr_weight": lr_weight,
        "projection_edit_strength": edit_strength,
        "projection_mask_blur_radius": float(mask_blur_radius),
        "before_mse_downsample_vs_lr": before["mse_downsample_vs_lr"],
        "before_psnr_downsample_vs_lr": before["psnr_downsample_vs_lr"],
        "after_mse_downsample_vs_lr": after["mse_downsample_vs_lr"],
        "after_psnr_downsample_vs_lr": after["psnr_downsample_vs_lr"],
        "psnr_gain_db": after["psnr_downsample_vs_lr"] - before["psnr_downsample_vs_lr"],
    }
    return projected, diagnostics


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
    mask_strategy: Optional[str] = None,
    semantic_refine_prompts: Optional[Sequence[str]] = None,
    semantic_protect_prompts: Optional[Sequence[str]] = None,
    semantic_clip_model_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    semantic_clip_device: str = "cpu",
    semantic_grid_size: int = 8,
    semantic_batch_size: int = 16,
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

    strategy = normalize_mask_strategy(mask_strategy or plan.mask_policy)
    semantic_prior_maps = None
    semantic_diagnostics: Dict[str, Any] = {}
    if strategy == "semantic_uncage":
        from .semantic_guidance import CLIPRegionPrior

        semantic_prior = CLIPRegionPrior(
            model_path=semantic_clip_model_path,
            device=semantic_clip_device,
        ).build_prior(
            init_image,
            refine_prompts=semantic_refine_prompts,
            protect_prompts=semantic_protect_prompts,
            grid_size=semantic_grid_size,
            batch_size=semantic_batch_size,
        )
        semantic_prior_maps = semantic_prior.maps()
        semantic_diagnostics = dict(semantic_prior.diagnostics)

    mask_image = adaptive_mask(
        init_image,
        plan,
        outpaint_mask=outpaint_mask,
        strategy=strategy,
        semantic_prior=semantic_prior_maps,
    )
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
    extra_paths = save_mask_strategy_artifacts(init_image, mask_image, output_dir, strategy, semantic_prior=semantic_prior_maps)

    mask_arr = np.asarray(mask_image, dtype=np.uint8) > 0
    score, maps = mask_score_map(init_image, strategy, semantic_prior=semantic_prior_maps)
    diagnostics: Dict[str, Any] = {
        **metrics,
        **semantic_diagnostics,
        "mode": plan.mode,
        "alpha": plan.alpha,
        "mask_strategy": strategy,
        "target_resolution": list(target_size),
        "masked_pixel_ratio": float(mask_arr.mean()),
        **mask_quality_diagnostics(mask_arr, score, maps, mask_budget_for_plan(plan), strategy),
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
        "paths": {**{key: str(value) for key, value in paths.items()}, **extra_paths},
    }
