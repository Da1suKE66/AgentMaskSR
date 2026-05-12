# Results

This file records command results, artifacts, metrics, and sync status for AgentMaskSR.

## Remote Safety Policy

All remote project outputs must stay under:

```text
/home/ma-user/workspace/llc/AgentSR
```

The project conda runtime is:

```text
/cache/llc/SR
```

## 2026-05-01 GitHub Sync

Status:

```text
initial upstream Meissonic baseline pushed
```

Target repository:

```text
https://github.com/Da1suKE66/AgentMaskSR
```

Notes:

- Remote `lsh-temp` clone succeeded.
- Remote HTTPS push failed because GitHub credentials were not available in the non-interactive SSH session.
- Local temporary clone push succeeded.

## 2026-05-01 Environment Setup

Requested environment path:

```text
/cache/llc/SR
```

Activation helper:

```bash
source envs/agentsr_cache_env.sh
```

The helper sets:

- `CONDA_PKGS_DIRS=/cache/llc/SR-pkgs`
- `PIP_CACHE_DIR=/cache/llc/SR-pip-cache`
- `HF_HOME=/home/ma-user/workspace/llc/AgentSR/.hf_cache`
- `MPLCONFIGDIR=/home/ma-user/workspace/llc/AgentSR/.mplconfig`

## 2026-05-01 Controller Dry-Run Outputs

Smoke-test command:

```bash
python tools/agent_mask_sr.py \
  --input_image assets/inpaint/0eKR4M2uuL8.jpg \
  --output_dir outputs/smoke_controller \
  --prompt "faithful super-resolution" \
  --mode sr \
  --target_resolution 512x512 \
  --dry_run
```

Outputs:

```text
outputs/smoke_controller/init_observation.png
outputs/smoke_controller/mask_refine.png
outputs/smoke_controller/agent_plan.json
outputs/smoke_controller/controller_metrics.json
```

Result:

```text
py_compile passed for agentsr/controller.py and tools/agent_mask_sr.py
dry-run controller generation passed
```

Recorded metrics:

```text
mse_downsample_vs_lr = 70.5918197631836
psnr_downsample_vs_lr = 29.643259832940526
masked_pixel_ratio = 0.315521240234375
target_resolution = 512x512
tile_grid = [{"x0": 0, "y0": 0, "x1": 512, "y1": 512}]
```

The dry run validates the controller and mask policy without downloading Meissonic checkpoints.

## 2026-05-01 Full Meissonic Pipeline Smoke Run

Status:

```text
completed
```

Environment:

```text
/cache/llc/SR
python 3.10.20
torch 2.8.0+cu128
torchvision 0.23.0
transformers 4.56.2
diffusers 0.35.1
accelerate 1.10.1
```

CUDA validation:

```text
torch.cuda.is_available() = True
GPU = NVIDIA A100-SXM4-80GB
```

Downloaded Hugging Face weights into the repository-local cache:

```text
/home/ma-user/workspace/llc/AgentSR/.hf_cache
```

Models loaded:

```text
MeissonFlow/Meissonic
laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

The first attempted 512x512 run failed in Meissonic's rotary-position path:

```text
RuntimeError: The size of tensor a (333) must match the size of tensor b (1101)
```

The first 1024x1024 run completed with fp16, but the decoded image was all black. The wrapper was then changed to default to `float32`, matching the official Meissonic scripts.

Successful command:

```bash
python tools/agent_mask_sr.py \
  --input_image assets/inpaint/0eKR4M2uuL8.jpg \
  --output_dir outputs/first_pipeline_run_1024_fp32 \
  --prompt "faithful super-resolution with clean texture detail" \
  --mode sr \
  --target_resolution 1024x1024 \
  --alpha 0.35 \
  --run_meissonic \
  --steps 4 \
  --guidance_scale 7.0 \
  --seed 66 \
  --dtype float32
```

Outputs:

```text
outputs/first_pipeline_run_1024_fp32/init_observation.png
outputs/first_pipeline_run_1024_fp32/mask_refine.png
outputs/first_pipeline_run_1024_fp32/agent_plan.json
outputs/first_pipeline_run_1024_fp32/controller_metrics.json
outputs/first_pipeline_run_1024_fp32/meissonic_refined.png
outputs/first_pipeline_run_1024_fp32/meissonic_metrics.json
```

Controller metrics:

```text
mode = sr
alpha = 0.35
target_resolution = 1024x1024
masked_pixel_ratio = 0.3082866668701172
mse_downsample_vs_lr = 48.346221923828125
psnr_downsample_vs_lr = 31.287178196100704
```

Refined output check:

```text
size = 1024x1024
RGB mean = [115.91, 110.99, 96.93]
RGB extrema = [(0, 255), (0, 249), (0, 255)]
```

Downsample consistency after Meissonic refinement:

```text
mse_downsample_vs_lr = 1477.1201171875
psnr_downsample_vs_lr = 16.436645479455237
```

Interpretation:

The first complete pipeline is now operational and produces a non-empty image. The low PSNR after Meissonic refinement shows the next research step clearly: add stronger observation-consistency scoring or token-level rejection so Meissonic's visual prior cannot drift too far from the LR observation.

## 2026-05-01 Observation Consistency Projection

Status:

```text
completed
```

Added a deterministic post-refinement projection in `agentsr/controller.py`:

1. Downsample the HR Meissonic candidate to the original LR size.
2. Compute the low-frequency residual against the LR observation.
3. Upsample the residual to HR and subtract it from the candidate.
4. Re-apply the controller mask so protected black-mask regions move back toward `init_observation.png`.

New output files:

```text
outputs/first_pipeline_run_1024_consistency/meissonic_consistent.png
outputs/first_pipeline_run_1024_consistency/consistency_projection_metrics.json
```

Successful integrated command:

```bash
python tools/agent_mask_sr.py \
  --input_image assets/inpaint/0eKR4M2uuL8.jpg \
  --output_dir outputs/first_pipeline_run_1024_consistency \
  --prompt "faithful super-resolution with clean texture detail" \
  --mode sr \
  --target_resolution 1024x1024 \
  --alpha 0.35 \
  --run_meissonic \
  --steps 4 \
  --guidance_scale 7.0 \
  --seed 66 \
  --dtype float32
```

Projection parameters:

```text
projection_steps = 2
projection_lr_weight = 0.5
projection_edit_strength = 0.87
projection_mask_blur_radius = 6.0
```

Metrics before projection:

```text
mse_downsample_vs_lr = 1477.1201171875
psnr_downsample_vs_lr = 16.436645479455237
```

Metrics after projection:

```text
mse_downsample_vs_lr = 120.35453796386719
psnr_downsample_vs_lr = 27.32617890889089
psnr_gain_db = 10.889533429435652
```

Non-copy check:

```text
mean_abs(meissonic_refined - init_observation) = [18.69, 18.47, 19.02]
mean_abs(meissonic_consistent - init_observation) = [3.42, 3.41, 3.49]
mean_abs(meissonic_consistent - meissonic_refined) = [15.27, 15.07, 15.53]
```

Interpretation:

The projection strongly improves LR consistency while retaining a controlled amount of Meissonic edit signal. It is still pixel-space, not token-level rejection, so the next method step is to move this residual score into candidate token acceptance or scheduler callbacks.


## 2026-05-12 Token+Rerank MVP Implementation

Status: implemented and dry-run validated.

Implemented changes:

- Fixed old absolute workspace path references from previous AgentMaskSR workspace path to /home/ma-user/workspace/llc/AgentSR in key docs and envs/agentsr_cache_env.sh.
- Added agentsr/token_masks.py for token-grid known, active, commit, and outpaint masks.
- Added agentsr/reranker.py for candidate LR L1, LR gradient L1, boundary L1, edit-penalty scoring, and score JSON output.
- Added agentsr/token_editor.py as a frozen Meissonic VQ-token editor wrapper exposing encode, decode, partial-mask refine, and candidate tokens.
- Updated tools/agent_mask_sr.py with mode token_rerank_sr.
- Projection is no longer default behavior. It only runs when --run_projection_ablation is explicitly passed and writes projection_ablation outputs.
- Added dry-run token metadata output: x_base_hr.png, round_00/token_masks.npz, round_00/active_token_mask.png, and run_summary.json.

Validation:

- python3 -m py_compile agentsr/*.py tools/*.py passed.
- Runtime imports for token_masks, reranker, and token_editor passed in existing /cache/llc/EditMGT environment.
- token_rerank_sr dry-run passed with assets/inpaint/0eKR4M2uuL8.jpg at 512x512.
- Dry-run output directory: outputs/token_rerank_dryrun.
- Dry-run token shape: 32 x 32.
- Dry-run active tokens: 330.
- Dry-run active ratio: 0.322265625.

Notes:

- Full Meissonic token-rerank smoke was not run because the documented /cache/llc/SR conda environment does not exist on lsh-stable and AgentSR has no local Hugging Face model cache yet.
- The implementation keeps full-run support in place through --mode token_rerank_sr --run_meissonic, but running it will require a valid Meissonic runtime and model cache under AgentSR.

## 2026-05-12 /cache/llc Runtime and HF Cache Setup

Status: completed.

Environment:

- Conda prefix: /cache/llc/SR
- Source environment: cloned from existing /cache/llc/EditMGT after direct PyTorch wheel download from download.pytorch.org stalled.
- Python: 3.10.20
- torch: 2.1.0+cu121
- CUDA reported by torch: 12.1
- torch.cuda.is_available(): True
- diffusers: 0.32.1
- transformers: 4.47.1
- HF cache: /cache/llc/SR-hf-cache
- Hub cache: /cache/llc/SR-hf-cache/hub
- Matplotlib cache: /cache/llc/SR-mplconfig

Model cache:

- Direct huggingface.co download timed out from lsh-stable.
- Download succeeded with HF_ENDPOINT=https://hf-mirror.com.
- Cached models: MeissonFlow/Meissonic and laion/CLIP-ViT-H-14-laion2B-s32B-b79K.
- Cache size after download: about 24G.

Environment script:

- envs/agentsr_cache_env.sh now activates /cache/llc/SR.
- HF_HOME now points to /cache/llc/SR-hf-cache.
- TRANSFORMERS_CACHE now points to /cache/llc/SR-hf-cache/hub so offline from_pretrained can see the downloaded snapshots.

Validation:

- Offline Meissonic pipeline load passed with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
- token_rerank_sr full smoke passed at 1024x1024 with 1 round, 1 candidate, 1 step.
- Smoke output directory: outputs/token_rerank_smoke_1024.
- Output files include final_hr.png, round_01/candidate_00.png, round_01/candidate_scores.json, and token_masks.npz.
- final_hr.png is non-black: RGB mean approximately [113.09, 109.15, 96.82], channel extrema all span 0 to 255.

Smoke command summary:

source envs/agentsr_cache_env.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/agent_mask_sr.py --input_image assets/inpaint/0eKR4M2uuL8.jpg --output_dir outputs/token_rerank_smoke_1024 --prompt faithful super-resolution --mode token_rerank_sr --target_resolution 1024x1024 --run_meissonic --rounds 1 --candidate_k 1 --steps 1 --guidance_scale 7.0 --seed 66 --dtype float32

Note:

- The smoke run is a functionality check, not a quality benchmark. It used only 1 candidate and 1 step, so the LR metric is expected to be poor.


## 2026-05-12 Real LR SR Test Flow and Conservative Mask Update

Status: completed.

Why changed:

- The previous smoke used the original 3927x3927 image as input and targeted 1024x1024, which was a downscale/refine test rather than super-resolution.
- A controlled 4x SR pair is now generated from the HR sample before running token_rerank_sr.

Added:

- tools/make_sr_test_input.py creates a deterministic LR/HR pair:
  - center-crop square source image
  - resize crop to HR reference
  - downsample HR reference to LR input
  - write sr_pair_metadata.json

Current test pair:

- Source: assets/inpaint/0eKR4M2uuL8.jpg, 3927x3927
- HR reference: outputs/sr_test_pair_dog_4x/hr_reference.png, 1024x1024
- LR input: outputs/sr_test_pair_dog_4x/lr_input.png, 256x256
- Scale factor: 4x

Mask strategy update:

- Previous SR mask budget: 0.10 + 0.30 * alpha, followed by MaxFilter(3). With alpha 0.35 this produced about 30% masked pixels after dilation.
- New SR mask budget: 0.06 + 0.18 * alpha, with no dilation for SR mode. With alpha 0.35 this targets about 12.3% masked pixels.
- Outpaint and sr_outpaint still keep dilation because boundary fill needs connected mask regions.

Real LR dry-run result:

- Output: outputs/token_rerank_real_lr_dryrun
- masked_pixel_ratio: 0.12315177917480469
- token_shape: 64x64
- active_token_ratio: 0.105712890625
- active_tokens: 433

Real LR smoke result:

- Output: outputs/token_rerank_real_lr_smoke_1024
- Run: 1024 target, 1 round, 1 candidate, 1 step, seed 66
- base_down_vs_lr: MSE 1.5036, L1 0.5343, PSNR 46.3594 dB
- final_down_vs_lr: MSE 328.0215, L1 8.2064, PSNR 22.9718 dB
- base_vs_hr_reference: PSNR 32.2725 dB
- final_vs_hr_reference: PSNR 22.3678 dB
- lr_worse_ratio: 0.994140625

Interpretation:

- The test flow is now a real 256->1024 SR setup.
- The mask is now much more conservative than before.
- The full smoke still accepts a bad candidate because candidate_k=1 and no reject/rollback policy is active yet.
- The next controller fix should reject candidates when LR consistency is worse than the base/current image by a threshold, even before the full agent loop exists.

## 2026-05-13 Candidate Reject/Rollback Smoke

Status: completed.

Change:

- Added default candidate rejection in token_rerank_sr.
- Projection remains off; rejection is based on observation scoring only.
- CLI controls added: --reject_lr_l1_margin, --reject_lr_worse_ratio, --disable_candidate_reject.

Run:

- Output: outputs/token_rerank_real_lr_reject_1024
- Input LR: outputs/sr_test_pair_dog_4x/lr_input.png, 256x256
- Target/final size: 1024x1024
- Run: 1 round, candidate_k=1, 1 step, seed 66, dtype float32.

Result:

- Current/base LR L1: 0.5343068242073059
- Candidate LR L1: 8.206395149230957
- Candidate LR grad L1: 2.9769833087921143
- Candidate boundary L1: 21.78203582763672
- Candidate edit penalty L1: 4.851805686950684
- Candidate worse ratio: 0.994140625
- Rejected: lr_l1_regression:7.672088>1.000000
- Final metrics: MSE 1.5036163330078125, PSNR 46.359433262838095 dB
- final_hr.png exactly matches x_base_hr.png: max abs diff 0, mean abs diff 0.0.
- Candidate image differs from base with mean abs diff 8.277235984802246 and is kept only under round_01 for inspection.

Interpretation:

- The controller no longer lets a single bad Meissonic candidate overwrite the main result.
- This is still a smoke/safety test, not a quality benchmark.
- Directory naming note: round_00 stores initial token masks; the first generated candidate is under round_01.

## 2026-05-13 Mask Strategy Sweep and Token Guard Checkpoint

Status: completed.

Goal alignment:

- Added MEISSONIC_SR_AGENT_GOAL.md as the standing project target for Meissonic-SR Agent.
- The mainline remains frozen Meissonic + partial token masks + candidate reranking + observation-based reject/remask.
- LR projection is still not used in the main result path.

Implemented:

- Added mask strategies: frequency, edge, variance, hybrid, uncage.
- Default controller mask policy is now uncage for conservative token-rerank experiments.
- Added mask strategy artifacts: mask_score.png, frequency_map.png, mask_overlay.png.
- Added tools/sweep_mask_strategies.py for dry-run strategy comparison.
- Added --mask_strategy and --token_mask_threshold to tools/agent_mask_sr.py.
- Added token_threshold to build_initial_token_masks.
- Added hard known-token merge in MeissonicTokenEditor.refine.
- Added hard known-pixel composite after VQ decode so known regions visually remain unchanged.
- Added active_lr_worse_ratio and stable_active_ratio to mask_update diagnostics.
- Added --reject_active_lr_worse_ratio; candidates with active local LR regression are rejected even when global regression is small.
- Added tools/vq_roundtrip.py to measure Meissonic VQ encode/decode drift.

Mask sweep:

- Output: outputs/mask_strategy_sweep_dog_4x_tok025
- Input: outputs/sr_test_pair_dog_4x/lr_input.png, 256x256 -> 1024x1024
- token_mask_threshold: 0.25

| strategy | pixel active | token active | detail top25 | edge top10 | texture top25 | flat leakage |
|---|---:|---:|---:|---:|---:|---:|
| frequency | 0.1232 | 0.1990 | 0.4787 | 1.0000 | 0.1232 | 0.0000 |
| edge | 0.1254 | 0.2083 | 0.4875 | 1.0000 | 0.1254 | 0.0000 |
| variance | 0.1232 | 0.1685 | 0.4124 | 0.6863 | 0.1232 | 0.0000 |
| hybrid | 0.1230 | 0.1965 | 0.4779 | 0.9115 | 0.1230 | 0.0000 |
| uncage | 0.1230 | 0.1160 | 0.2949 | 0.4804 | 0.1230 | 0.0810 |

Interpretation:

- frequency/edge/hybrid over-target strong edges and create broad structural edit risk.
- uncage is more conservative at token level and keeps strongest edge cages more frozen, so it is a better first conservative_sr default.
- token_mask_threshold matters: with threshold 0.5, uncage activates 93 tokens; with threshold 0.25, it activates 475 tokens.

VQ round-trip finding:

- Output: outputs/vq_roundtrip_dog_lr_to_1024
- base input downsample vs LR PSNR: 46.3594 dB
- VQ recon downsample vs LR PSNR: 31.9810 dB
- VQ roundtrip image PSNR: 31.8969 dB
- VQ roundtrip L1: 5.0024
- recon LR L1: 4.9597

Interpretation:

- Meissonic VQ decode alone can break LR consistency, so token-level known merge is not enough to visually preserve known regions.
- Hard known-pixel composite is necessary observation protection, not LR projection.

Smoke results:

- uncage + threshold 0.25 + pixel guard reduced global candidate mean abs from 7.7904 to 3.7865 and known mean abs to 0.0, but candidate LR L1 was still 4.0802 and was rejected.
- uncage + threshold 0.5 + pixel guard reduced candidate LR L1 to 0.9940, but active_lr_worse_ratio was 1.0 and active tokens could not commit, so the stricter active-local reject now correctly rejects it.
- candidate_k=2 smoke selected seed 67 over seed 66: total 4.9610 vs 5.0471, but it was still rejected by active_lr_worse_ratio=1.0.

Next:

- Add candidate/metric sweep over mask_strategy, token_mask_threshold, strength, guidance, and candidate_k.
- Add optional CLIP/VLM-style semantic scoring so evaluation is not only dB/LR error.
- Investigate active-token acceptance: current Meissonic samples improve visual detail but locally worsen LR in all active tokens for this dog test.
