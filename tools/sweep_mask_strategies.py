#!/usr/bin/env python
"""Dry-run and compare Meissonic-SR token mask strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentsr.controller import MASK_STRATEGIES, build_refinement_assets, derive_agent_plan  # noqa: E402
from agentsr.token_masks import build_initial_token_masks, mask_metadata, save_token_masks_npz, token_mask_to_pixel_mask  # noqa: E402

DEFAULT_SWEEP_STRATEGIES = [strategy for strategy in MASK_STRATEGIES if strategy != "semantic_uncage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare deterministic SR mask strategies without running Meissonic.")
    parser.add_argument("--input_image", required=True, help="LR observation image.")
    parser.add_argument("--output_dir", default="outputs/mask_strategy_sweep", help="Repository-local output directory.")
    parser.add_argument("--prompt", default="faithful super-resolution")
    parser.add_argument("--target_resolution", default="1024x1024")
    parser.add_argument("--mode", default="sr", choices=["sr", "detail", "outpaint", "sr_outpaint"])
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--token_vae_scale_factor", type=int, default=16)
    parser.add_argument("--token_mask_threshold", type=float, default=0.25)
    parser.add_argument("--strategies", nargs="*", choices=list(MASK_STRATEGIES), default=DEFAULT_SWEEP_STRATEGIES)
    parser.add_argument("--semantic_refine_prompts", default=None)
    parser.add_argument("--semantic_protect_prompts", default=None)
    parser.add_argument("--semantic_clip_model_path", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--semantic_clip_device", default="cpu")
    parser.add_argument("--semantic_grid_size", type=int, default=8)
    parser.add_argument("--semantic_batch_size", type=int, default=16)
    parser.add_argument("--outpaint_direction", nargs="*", default=None, choices=["left", "right", "top", "bottom"])
    parser.add_argument("--outpaint_margin_ratio", type=float, default=0.18)
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--tile_overlap", type=int, default=128)
    return parser.parse_args()


def ensure_repo_local(path: Path) -> Path:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if repo not in (resolved, *resolved.parents):
        raise ValueError(f"output path must stay inside repository: {repo}")
    return resolved


def _contact_sheet(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    thumb_w, thumb_h = 256, 256
    label_h = 62
    pad = 12
    cols = min(3, len(rows))
    grid_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, grid_rows * (thumb_h + label_h) + (grid_rows + 1) * pad), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(rows):
        col = idx % cols
        grid_y = idx // cols
        x = pad + col * (thumb_w + pad)
        y = pad + grid_y * (thumb_h + label_h + pad)
        overlay = Image.open(row["paths"]["mask_overlay"]).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.BICUBIC)
        sheet.paste(overlay, (x, y))
        meta = row["token_masks"]
        diag = row["diagnostics"]
        label = (
            f"{row['strategy']}  active={meta['active_ratio']:.3f}\n"
            f"detail={diag['detail_coverage_top25']:.3f} edge={diag['edge_coverage_top10']:.3f}\n"
            f"texture={diag['texture_coverage_top25']:.3f} flat={diag['flat_leakage_bottom25']:.3f}"
        )
        draw.multiline_text((x, y + thumb_h + 6), label, fill=(20, 20, 20), spacing=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_markdown(path: Path, rows: List[Dict[str, Any]], contact_sheet: Path) -> None:
    lines = [
        "# Mask Strategy Sweep",
        "",
        f"Contact sheet: `{contact_sheet}`",
        "",
        "| strategy | pixel active | token active | detail top25 | edge top10 | texture top25 | flat leakage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        diag = row["diagnostics"]
        meta = row["token_masks"]
        lines.append(
            "| {strategy} | {pixel:.4f} | {token:.4f} | {detail:.4f} | {edge:.4f} | {texture:.4f} | {flat:.4f} |".format(
                strategy=row["strategy"],
                pixel=diag["masked_pixel_ratio"],
                token=meta["active_ratio"],
                detail=diag["detail_coverage_top25"],
                edge=diag["edge_coverage_top10"],
                texture=diag["texture_coverage_top25"],
                flat=diag["flat_leakage_bottom25"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "",
            "- `frequency` is the original high-frequency/variance baseline.",
            "- `edge` stresses structural lines and is useful as an ablation, but can over-edit object contours.",
            "- `variance` stresses textured regions and tends to avoid simple edges.",
            "- `hybrid` mixes frequency, variance, edge and a small deterministic sparse field.",
            "- `uncage` freezes strongest structural edge cages and releases sparse texture/detail islands around them.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = ensure_repo_local(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    observation = Image.open(args.input_image).convert("RGB")
    rows: List[Dict[str, Any]] = []

    for strategy in args.strategies:
        plan = derive_agent_plan(
            args.prompt,
            target_resolution=args.target_resolution,
            mode=args.mode,
            alpha=args.alpha,
            outpaint_direction=args.outpaint_direction,
        )
        plan.mask_policy = strategy
        strategy_dir = output_dir / strategy
        assets = build_refinement_assets(
            observation,
            plan,
            strategy_dir,
            outpaint_margin_ratio=args.outpaint_margin_ratio,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            mask_strategy=strategy,
            semantic_refine_prompts=args.semantic_refine_prompts,
            semantic_protect_prompts=args.semantic_protect_prompts,
            semantic_clip_model_path=args.semantic_clip_model_path,
            semantic_clip_device=args.semantic_clip_device,
            semantic_grid_size=args.semantic_grid_size,
            semantic_batch_size=args.semantic_batch_size,
        )
        masks = build_initial_token_masks(
            assets["mask_image"],
            plan.target_resolution,
            vae_scale_factor=args.token_vae_scale_factor,
            token_threshold=args.token_mask_threshold,
        )
        save_token_masks_npz(strategy_dir / "token_masks.npz", masks)
        token_mask_to_pixel_mask(masks.active, plan.target_resolution).save(strategy_dir / "active_token_mask.png")

        row = {
            "strategy": strategy,
            "output_dir": str(strategy_dir),
            "paths": assets["paths"],
            "diagnostics": assets["diagnostics"],
            "token_masks": mask_metadata(masks),
        }
        with (strategy_dir / "mask_strategy_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        rows.append(row)

    summary = {
        "input_image": str(Path(args.input_image)),
        "output_dir": str(output_dir),
        "target_resolution": args.target_resolution,
        "mode": args.mode,
        "alpha": args.alpha,
        "token_mask_threshold": args.token_mask_threshold,
        "strategies": rows,
    }
    summary_path = output_dir / "mask_strategy_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    contact_sheet = output_dir / "mask_strategy_contact_sheet.png"
    _contact_sheet(rows, contact_sheet)
    _write_markdown(output_dir / "mask_strategy_summary.md", rows, contact_sheet)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
