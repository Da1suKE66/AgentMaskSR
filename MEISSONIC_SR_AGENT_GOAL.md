# Meissonic-SR Agent Goal

This file is the standing project target. Every implementation checkpoint should be checked against it before committing.

## Core Direction

Build **Meissonic-SR Agent** as a training-free, observation-constrained super-resolution framework:

- Meissonic is frozen and only generates masked VQ/image tokens.
- The controller, analyzer, and later the agent decide where to mask, how strongly to sample, which candidates to keep, and which tokens to commit or remask.
- The low-resolution image `y_lr` is an observation used for deterministic upscale, candidate scoring, rejection/remask, and commit decisions.
- LR projection is not the main result path. It may exist only as an explicit ablation and must not overwrite the main output.

## Target Pipeline

```text
Input LR image y_lr
  -> deterministic upscale x_base_hr
  -> Meissonic VQ encode z_base
  -> analyzer/controller builds semantic/degradation/risk evidence
  -> token masks: known / active / commit / outpaint
  -> Meissonic partial-mask refinement
  -> K candidate HR images and token ids
  -> LR + boundary + semantic scoring
  -> select best candidate
  -> commit stable tokens, remask bad tokens
  -> repeat 2-4 rounds
  -> final HR image
```

Key rule: **Meissonic does not decide what to change. The controller or agent decides what to change.**

## Current Mainline

Keep the first reliable mainline small:

```text
Frozen Meissonic
+ partial-mask token editing
+ LR/boundary candidate scoring
+ reject/remask safety
```

Only after this is stable, add:

```text
+ understanding-guided semantic masks
+ agentic planning loop
+ early commit
+ tile-global 2048+ decoding
```

## Mask Controller Requirements

The mask controller is the core system component.

Token update score should move toward:

```text
S_i =
  w_freq   * F_i   # high-frequency / edge / texture evidence
+ w_sem    * A_i   # semantic priority from VLM/analyzer
+ w_lr     * L_i   # local LR consistency error
+ w_ent    * H_i   # Meissonic token confidence / entropy, when available
+ w_bd     * B_i   # boundary or tile overlap risk
+ w_pos    * P_i   # outpaint/border prior
- w_commit * C_i   # already committed tokens stay frozen
```

MVP mask work should compare multiple deterministic strategies instead of assuming one is correct:

- `frequency`: original gradient + local variance detail mask.
- `edge`: structural edge ablation.
- `variance`: texture-heavy mask.
- `hybrid`: mixed evidence plus sparse deterministic sampling.
- `uncage`: freeze strongest structural edge cages while releasing sparse texture/detail islands.

## Modes

Start with discrete modes, not continuous alpha-only behavior.

| mode | goal | rough mask ratio | strength | candidate K |
|---|---|---:|---:|---:|
| conservative_sr | preserve structure, add small truthful detail | 10-25% | 0.15-0.30 | 1-2 |
| perceptual_sr | richer texture without layout changes | 25-45% | 0.30-0.50 | 2-4 |
| outpaint / uncrop | generate outside, preserve inside | outside mostly active | mode-specific | 4 |

Alpha can later compress these modes, but the first robust implementation should expose and test the discrete modes.

## Candidate Scoring

Candidate selection must not use beauty alone. It should reject hallucinations that break the LR observation.

Scoring target:

```text
Score(x_k) =
  lambda_lr   * LR L1(D(x_k), y_lr)
+ lambda_grad * LR gradient L1
+ lambda_bd   * boundary discontinuity
+ lambda_id   * semantic or identity drift
- lambda_nat  * naturalness / quality score
- lambda_txt  * prompt or caption alignment
```

MVP scoring already includes LR L1, LR gradient L1, boundary L1, and known/commit edit penalty. Next work should add optional semantic/multimodal evaluation rather than relying only on dB.

Accept/reject rules:

- Reject if LR error regresses beyond the threshold.
- Reject if local LR worse ratio is too high.
- Reject or remask if text/face/identity checks fail once semantic analyzers exist.
- Commit only tokens with agreement, stable LR error, stable boundary error, and semantic pass.

## Agent Role

The agent is a stage-level planner, not a token sampler.

It should call tools in this order:

```text
analyze()
build_mask()
generate_candidates()
score_candidates()
commit_or_remask()
```

Per round, it should output a plan similar to:

```json
{
  "mode": "conservative_sr",
  "mask_strategy": "uncage",
  "mask_ratio": 0.18,
  "strength": 0.25,
  "guidance_scale": 7.0,
  "candidate_k": 2,
  "regions_to_refine": ["fur", "grass", "building edges"],
  "regions_to_freeze": ["text", "face center", "flat sky"]
}
```

## Implementation Order

1. Meissonic img2img / inpaint baseline.
2. VQ round-trip test.
3. Partial mask SR.
4. LR consistency accept/reject.
5. Mask strategy sweep: frequency, edge, variance, hybrid, uncage.
6. Candidate reranking with K=2 then K=4.
7. Optional CLIP/VLM-based semantic and multimodal metrics.
8. Stage-level agent loop.
9. Early commit.
10. 2048 tile-global decoding.

## Experiment Discipline

- Use true SR test pairs, not accidental downscale/refine tests.
- Record original LR size, target HR size, mask ratio, token ratio, candidate metrics, accept/reject reason, and final artifact paths.
- Keep `/cache/llc` for env/model caches and keep large outputs out of git.
- Commit and push every meaningful implementation or experiment checkpoint.
