# Meissonic-SR Agent Research Notes

## Goal

Main direction:

```text
LR observation
-> progressive 2x deterministic lift
-> Meissonic masked VQ-token refinement
-> candidate reranking by observation consistency
-> commit/remask planning
```

The single-stage `token_rerank_sr` path is kept as an ablation. The main path is now `progressive_token_rerank_sr`.

## Meissonic / MGIT Notes

Sources:

- Meissonic paper: https://arxiv.org/abs/2410.08261
- Hugging Face paper page: https://huggingface.co/papers/2410.08261
- Official code: https://github.com/viiika/Meissonic
- Official model: https://huggingface.co/MeissonFlow/Meissonic

Relevant properties for SR:

- Meissonic is a non-autoregressive masked image model over VQ tokens.
- Editing works by encoding the input image to discrete tokens, masking selected tokens, predicting them iteratively, and decoding back to pixels.
- The paper emphasizes variable mask ratios and mask-rate conditioning. This matters for SR because a tiny partial mask is not just a spatial operation; it changes the sampler's starting mask ratio and therefore the denoising dynamics.
- The scheduler uses confidence plus Gumbel top-k remasking. Temperature directly affects which low-confidence tokens stay masked across iterations.
- The official paper reports inference with CFG 9 and 48 steps, while early AgentSR smoke tests used 8 steps and low effective strength, often giving only 1 actual sampling step.
- The implementation uses an inpaint-specific linear masking schedule. This makes `strength * steps` a critical knob: low strength with few steps collapses to almost no iterative refinement.

Implications:

- We should not treat Meissonic as a generic pixel detail enhancer.
- Stage 1 should be very conservative because any identity/structure drift gets amplified in later stages.
- Candidate reranking is more important than single sampling because Gumbel sampling can produce materially different local edits.
- Semantic/VLM priors should mostly protect or rerank same-budget choices; direct semantic mask expansion is unsafe.

## Experiments

Input:

- `outputs/sr_test_pair_dog_4x/lr_input.png`
- LR size: 256x256
- Target: 1024x1024

Generated summary artifacts:

- `outputs/progressive_experiment_summary.json`
- `outputs/progressive_experiment_summary.md`

| run | thresholds | strengths | steps | temp | active | stage1 L1 | stage2 orig L1 | final PSNR |
|---|---|---|---|---|---:|---:|---:|---:|
| progressive_exp_stage1_ultra_stage2_conservative_k2 | [0.7, 0.6] | [0.18, 0.16] | [24, 24] | [0.5, 0.0] | [6, 11] | 0.7828 | 0.8370 | 37.3643 |
| progressive_exp_stage1_ultraconservative_k2 | [0.7, 0.55] | [0.18, 0.18] | [24, 24] | [0.5, 0.0] | [6, 31] | 0.7828 | 0.9036 | 36.5978 |
| progressive_exp_best_round2_k2 | [0.7, 0.6] | [0.18, 0.16] | [24, 24] | [0.5, 0.0] | [6, 12] | 0.8232 | 0.9156 | 34.8687 |
| progressive_exp_conservative_k2 | [0.6, 0.55] | [0.2, 0.18] | [24, 24] | [0.5, 0.0] | [20, 17] | 1.3003 | 1.3474 | 31.3596 |
| progressive_exp_conservative_lowtemp | [0.6, 0.55] | [0.2, 0.18] | [24, 24] | [0.5, 0.0] | [20, 17] | 1.3248 | 1.3706 | 30.6789 |
| progressive_exp_meissonic_like_temp | [0.6, 0.55] | [0.25, 0.2] | [48, 48] | [2.0, 0.0] | [20, 20] | 1.3137 | 1.4082 | 30.5516 |
| progressive_uncage_dog_k1_smoke_fixed | default old | default old | default old | default old | [46, 18] | 2.1129 | 2.0922 | 28.3105 |

## Findings

- Progressive 2x is runnable after fixing non-1024 positional ids in `src/pipeline_inpaint.py`.
- The first successful progressive smoke was too aggressive at stage 1 and damaged the dog eye/face.
- Stage 1 active budget dominates quality. Reducing stage 1 from 46 active tokens to 6 improved final PSNR from 28.31 to 37.36.
- Candidate reranking helps. With the same conservative setup, K=2 improved over K=1.
- More Meissonic-like 48-step/high-temperature sampling did not help on this SR observation task. It can increase generative freedom, which is not always good for observation-constrained SR.
- Two rounds currently worsens the result. The remask/commit loop still lacks a proper stable-token acceptance policy.
- Semantic_uncage currently expands stage-2 active tokens too much. It should be redesigned as same-budget semantic selection or hard-protect freezing, not direct active-mask expansion.

## Current Recommended Progressive Defaults

These are now used when progressive mode is called without stage overrides:

```text
stage_token_mask_thresholds = 0.70,0.60
stage_refine_strengths = 0.18,0.16
stage_guidance_scales = 3.0,4.0
stage_steps = 24,24
temperature_start = 0.5
temperature_end = 0.0
rounds = 1
candidate_k = user-controlled, use 2 for quality checks
```

## Next Work

- Implement same-budget semantic token selection:
  - start from the same number of uncage token candidates;
  - suppress high protect-score tokens;
  - promote refine-score tokens only within the same token budget.
- Add a hard-protect map for eyes/nose/text/object boundary before token selection.
- Add original-LR-aware reranking weight in progressive scoring, not only logging.
- Make commit require actual local improvement, not just no global rejection.
- Run K=4 only on small active budgets.
