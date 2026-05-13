#!/usr/bin/env python
"""Summarize progressive token-rerank SR experiment folders."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_round(stage: Dict[str, Any]) -> Dict[str, Any]:
    rounds = stage.get("rounds") or []
    return rounds[-1] if rounds else {}


def _row(summary_path: Path) -> Dict[str, Any]:
    summary = _load_summary(summary_path)
    stages = summary.get("stages") or []
    first = stages[0] if stages else {}
    last = stages[-1] if stages else {}
    first_round = _last_round(first)
    last_round = _last_round(last)
    return {
        "run": summary_path.parent.name,
        "summary_path": str(summary_path),
        "mode": summary.get("mode"),
        "thresholds": summary.get("stage_overrides", {}).get("token_mask_thresholds"),
        "strengths": summary.get("stage_overrides", {}).get("refine_strengths"),
        "guidance": summary.get("stage_overrides", {}).get("guidance_scales"),
        "steps": summary.get("stage_overrides", {}).get("steps"),
        "temperature": summary.get("stage_overrides", {}).get("temperature"),
        "stage_active_tokens": [
            stage.get("initial_token_masks", {}).get("active_tokens")
            for stage in stages
        ],
        "stage1_candidate_ref_l1": first_round.get("candidate_stage_ref_l1"),
        "stage2_candidate_ref_l1": last_round.get("candidate_stage_ref_l1"),
        "stage2_candidate_original_lr_l1": last_round.get("candidate_original_lr_l1"),
        "final_psnr": summary.get("final_metrics", {}).get("psnr_downsample_vs_lr"),
        "final_mse": summary.get("final_metrics", {}).get("mse_downsample_vs_lr"),
        "accepted_rounds": sum(
            1
            for stage in stages
            for round_item in stage.get("rounds", [])
            if round_item.get("accepted")
        ),
        "reject_reasons": [
            round_item.get("reject_reason")
            for stage in stages
            for round_item in stage.get("rounds", [])
            if round_item.get("reject_reason")
        ],
    }


def _find_summaries(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            if path.is_dir():
                path = path / "run_summary.json"
            if path.exists() and path.name == "run_summary.json":
                paths.append(path)
    return sorted(set(paths))


def _write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Progressive Token-Rerank Experiments",
        "",
        "| run | thresholds | strengths | steps | temp | active | stage1 L1 | stage2 orig L1 | final PSNR | accepted |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {thresholds} | {strengths} | {steps} | {temperature} | {active} | {s1:.4f} | {s2:.4f} | {psnr:.4f} | {accepted} |".format(
                run=row["run"],
                thresholds=row.get("thresholds"),
                strengths=row.get("strengths"),
                steps=row.get("steps"),
                temperature=row.get("temperature"),
                active=row.get("stage_active_tokens"),
                s1=float(row.get("stage1_candidate_ref_l1") or 0.0),
                s2=float(row.get("stage2_candidate_original_lr_l1") or 0.0),
                psnr=float(row.get("final_psnr") or 0.0),
                accepted=row.get("accepted_rounds", 0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize progressive token-rerank SR runs.")
    parser.add_argument(
        "patterns",
        nargs="*",
        default=[
            "outputs/progressive_uncage_dog_k1_smoke_fixed",
            "outputs/progressive_exp_*",
        ],
        help="Run directories or glob patterns pointing to run_summary.json files.",
    )
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    rows = [_row(path) for path in _find_summaries(args.patterns)]
    rows = [row for row in rows if row.get("mode") == "progressive_token_rerank_sr"]
    rows.sort(key=lambda item: float(item.get("final_psnr") or 0.0), reverse=True)

    payload = {"runs": rows}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        _write_markdown(Path(args.output_md), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
