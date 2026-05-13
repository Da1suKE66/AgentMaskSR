#!/usr/bin/env python
"""Run a small token-rerank SR parameter sweep and aggregate results."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _csv_floats(values: str) -> List[float]:
    return [float(item) for item in values.split(",") if item.strip()]


def _csv_ints(values: str) -> List[int]:
    return [int(item) for item in values.split(",") if item.strip()]


def _csv_strings(values: str) -> List[str]:
    return [item.strip() for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep token-rerank SR parameters.")
    parser.add_argument("--input_image", required=True)
    parser.add_argument("--output_dir", default="outputs/token_rerank_sweep")
    parser.add_argument("--prompt", default="faithful super-resolution")
    parser.add_argument(
        "--mode",
        default="token_rerank_sr",
        choices=["token_rerank_sr", "progressive_token_rerank_sr", "cascaded_token_rerank_sr"],
    )
    parser.add_argument("--target_resolution", default="1024x1024")
    parser.add_argument("--strategies", default="uncage")
    parser.add_argument("--token_thresholds", default="0.5")
    parser.add_argument("--strengths", default="0.15,0.25")
    parser.add_argument("--guidance_scales", default="4.0,5.0")
    parser.add_argument("--candidate_ks", default="1,2")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_meissonic", action="store_true")
    parser.add_argument("--enable_clip_score", action="store_true")
    parser.add_argument("--clip_text_weight", type=float, default=0.0)
    parser.add_argument("--clip_image_weight", type=float, default=0.0)
    parser.add_argument("--clip_score_device", default="cpu")
    parser.add_argument("--semantic_refine_prompts", default=None)
    parser.add_argument("--semantic_protect_prompts", default=None)
    parser.add_argument("--semantic_clip_model_path", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    parser.add_argument("--semantic_clip_device", default="cpu")
    parser.add_argument("--semantic_grid_size", type=int, default=8)
    parser.add_argument("--semantic_batch_size", type=int, default=16)
    parser.add_argument("--max_runs", type=int, default=None)
    return parser.parse_args()


def ensure_repo_local(path: Path) -> Path:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if repo not in (resolved, *resolved.parents):
        raise ValueError(f"output path must stay inside repository: {repo}")
    return resolved


def _iter_configs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    strategies = _csv_strings(args.strategies)
    thresholds = _csv_floats(args.token_thresholds)
    strengths = _csv_floats(args.strengths)
    guidance = _csv_floats(args.guidance_scales)
    candidate_ks = _csv_ints(args.candidate_ks)
    iterator = itertools.product(strategies, thresholds, strengths, guidance, candidate_ks)
    for idx, (strategy, threshold, strength, guidance_scale, candidate_k) in enumerate(iterator):
        if args.max_runs is not None and idx >= args.max_runs:
            break
        yield {
            "strategy": strategy,
            "token_threshold": threshold,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "candidate_k": candidate_k,
        }


def _run_config(args: argparse.Namespace, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    run_name = (
        f"{args.mode}_{config['strategy']}_tok{config['token_threshold']:.2f}_"
        f"str{config['strength']:.2f}_gs{config['guidance_scale']:.1f}_k{config['candidate_k']}"
    ).replace(".", "p")
    run_dir = output_dir / run_name
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "agent_mask_sr.py"),
        "--input_image",
        args.input_image,
        "--output_dir",
        str(run_dir),
        "--prompt",
        args.prompt,
        "--mode",
        args.mode,
        "--target_resolution",
        args.target_resolution,
        "--mask_strategy",
        config["strategy"],
        "--token_mask_threshold",
        str(config["token_threshold"]),
        "--rounds",
        str(args.rounds),
        "--candidate_k",
        str(config["candidate_k"]),
        "--steps",
        str(args.steps),
        "--guidance_scale",
        str(config["guidance_scale"]),
        "--refine_strength",
        str(config["strength"]),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    if args.semantic_refine_prompts is not None:
        cmd.extend(["--semantic_refine_prompts", args.semantic_refine_prompts])
    if args.semantic_protect_prompts is not None:
        cmd.extend(["--semantic_protect_prompts", args.semantic_protect_prompts])
    cmd.extend(
        [
            "--semantic_clip_model_path",
            args.semantic_clip_model_path,
            "--semantic_clip_device",
            args.semantic_clip_device,
            "--semantic_grid_size",
            str(args.semantic_grid_size),
            "--semantic_batch_size",
            str(args.semantic_batch_size),
        ]
    )
    if args.run_meissonic:
        cmd.append("--run_meissonic")
    else:
        cmd.append("--dry_run")
    if args.enable_clip_score:
        cmd.extend(
            [
                "--enable_clip_score",
                "--clip_score_device",
                args.clip_score_device,
                "--clip_text_weight",
                str(args.clip_text_weight),
                "--clip_image_weight",
                str(args.clip_image_weight),
            ]
        )

    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    summary_path = run_dir / "run_summary.json"
    result: Dict[str, Any] = {
        "run_name": run_name,
        "output_dir": str(run_dir),
        "config": config,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result["summary_path"] = str(summary_path)
        if args.mode in {"progressive_token_rerank_sr", "cascaded_token_rerank_sr"}:
            stages = summary.get("stages") or []
            last_stage = stages[-1] if stages else {}
            result["stages"] = [
                {
                    "stage": stage.get("stage"),
                    "target_resolution": stage.get("target_resolution"),
                    "initial_token_masks": stage.get("initial_token_masks"),
                    "stage_final_source": stage.get("stage_final_source"),
                    "stage_consistency_metrics": stage.get("stage_consistency_metrics"),
                    "original_lr_metrics": stage.get("original_lr_metrics"),
                }
                for stage in stages
            ]
            result["initial_token_masks"] = last_stage.get("initial_token_masks")
            rounds = last_stage.get("rounds") or []
            score_dir = Path(last_stage.get("stage_final", run_dir)).parent
        else:
            result["initial_token_masks"] = summary.get("initial_token_masks")
            rounds = summary.get("rounds") or []
            score_dir = run_dir
        result["final_metrics"] = summary.get("final_metrics")
        if rounds:
            last_round = rounds[-1]
            result["accepted"] = last_round.get("accepted")
            result["reject_reason"] = last_round.get("reject_reason")
            result["best_candidate"] = last_round.get("best_candidate")
            result["best_score"] = last_round.get("best_score")
            result["candidate_lr_l1"] = last_round.get("candidate_lr_l1", last_round.get("candidate_stage_ref_l1"))
            result["candidate_lr_grad_l1"] = last_round.get(
                "candidate_lr_grad_l1",
                last_round.get("candidate_stage_ref_grad_l1"),
            )
            result["candidate_original_lr_l1"] = last_round.get("candidate_original_lr_l1")
            result["mask_update"] = last_round.get("mask_update")
            score_path = score_dir / f"round_{int(last_round.get('round', 1)):02d}" / "candidate_scores.json"
            if score_path.exists():
                score_payload = json.loads(score_path.read_text(encoding="utf-8"))
                selected = next(
                    (item for item in score_payload.get("candidates", []) if item.get("selected")),
                    None,
                )
                result["candidate_scores_path"] = str(score_path)
                result["selected_candidate_score"] = selected
    else:
        result["stdout_tail"] = completed.stdout[-2000:]
    return result


def _write_markdown(path: Path, results: List[Dict[str, Any]]) -> None:
    lines = [
        "# Token Rerank Sweep",
        "",
        "| run | strategy | tok thr | strength | guidance | K | active | accepted | lr_l1 | score | CLIP txt | CLIP img | reject |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        config = item["config"]
        masks = item.get("initial_token_masks") or {}
        mask_update = item.get("mask_update") or {}
        reject = item.get("reject_reason") or ""
        selected = item.get("selected_candidate_score") or {}
        if len(reject) > 42:
            reject = reject[:39] + "..."
        lines.append(
            "| {run} | {strategy} | {thr:.2f} | {strength:.2f} | {guidance:.1f} | {k} | {active:.4f} | {accepted} | {lr:.4f} | {score:.4f} | {clip_txt:.4f} | {clip_img:.4f} | {reject} |".format(
                run=item["run_name"],
                strategy=config["strategy"],
                thr=config["token_threshold"],
                strength=config["strength"],
                guidance=config["guidance_scale"],
                k=config["candidate_k"],
                active=float(masks.get("active_ratio", 0.0)),
                accepted=item.get("accepted"),
                lr=float(item.get("candidate_lr_l1") or 0.0),
                score=float(item.get("best_score") or 0.0),
                clip_txt=float(selected.get("clip_text_similarity") or 0.0),
                clip_img=float(selected.get("clip_image_similarity") or 0.0),
                reject=reject,
            )
        )
        if mask_update:
            lines.append(
                "<!-- {run}: active_lr_worse_ratio={active_worse}, stable_active_ratio={stable} -->".format(
                    run=item["run_name"],
                    active_worse=mask_update.get("active_lr_worse_ratio"),
                    stable=mask_update.get("stable_active_ratio"),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = ensure_repo_local(Path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for config in _iter_configs(args):
        print(f"[sweep] {config}", flush=True)
        results.append(_run_config(args, output_dir, config))

    payload = {
        "input_image": args.input_image,
        "output_dir": str(output_dir),
        "mode": args.mode,
        "run_meissonic": bool(args.run_meissonic),
        "enable_clip_score": bool(args.enable_clip_score),
        "results": results,
    }
    (output_dir / "sweep_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(output_dir / "sweep_summary.md", results)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
