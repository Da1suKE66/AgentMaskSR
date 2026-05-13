# Meissonic-SR Agent Progress

更新时间：2026-05-12
远端：`lsh-stable`
工作目录：`/home/ma-user/workspace/llc/AgentSR`

## 约束

- 本次检查和记录只针对 `/home/ma-user/workspace/llc/AgentSR`。
- 后续代码、日志、实验输出、Markdown 记录都必须写在该目录内。
- 当前仓库已有文档和环境脚本仍多处引用旧路径 `previous AgentMaskSR workspace path`。在运行环境脚本或实验前，应先在本仓库内修正这些路径，避免写到工作目录外。
- 本次检查阶段未修改代码文件，只新增本 Markdown 进展记录。

## Meissonic-SR Agent 落地计划

### Phase 0：路径与基线安全

1. 将 README、PLAN、RESULTS、HISTORY、CHAT、`envs/agentsr_cache_env.sh` 中的旧工作目录统一到 `/home/ma-user/workspace/llc/AgentSR`。
2. 保持缓存和输出路径显式可控，确保 Hugging Face、Matplotlib、实验输出不落到工作目录外。
3. 跑 `py_compile` 和 controller dry-run，确认当前代码在 AgentSR 路径下仍可执行。

### Phase 1：官方 Meissonic 能力确认

1. 跑通 text-to-image、image-to-image、inpainting 三个官方/现有入口。
2. 重点确认 1024x1024 inpaint：白色 mask 重绘、黑色区域保持。
3. 固定安全 dtype、steps、guidance、seed 组合，避免 fp16 黑图问题复现。

### Phase 2：VQ round-trip

1. 新增 `vq_roundtrip.py`：`x_hr -> VQ encode -> VQ decode -> x_recon`。
2. 记录颜色漂移、边缘模糊、文字破坏和 LR downsample PSNR。
3. 如果 VQ round-trip 自身破坏结构，需要先把 controller 的保真目标降到 VQ 可达范围。

### Phase 3：Partial-mask SR MVP

1. 保留现有 bicubic/upscale 初始化。
2. 用 frequency/edge/detail map 生成 active mask，平坦区域 freeze。
3. 调 Meissonic inpaint 只重采样 active 区域。
4. 输出 `init_observation.png`、`mask_refine.png`、`meissonic_refined.png`、`controller_metrics.json`。

### Phase 4：LR consistency accept/reject

1. 不只做 pixel-space projection，还要做候选接受/拒绝。
2. 对每个候选计算 `D(x_candidate) - y_lr` 的 L1/MSE/PSNR 和梯度误差。
3. 如果全局或局部 LR error 变差，拒绝候选或把局部区域 remask。
4. 输出候选评分 JSON，形成 observation-constrained SR 证据链。

### Phase 5：candidate reranking

1. 先实现 K=2，仅在 active mask 面积较小或高风险区域启用。
2. 简化评分：`-LR_error - boundary_error + CLIP/prompt_similarity`。
3. 后续扩展到 K=4 和 selected windows rerank。

### Phase 6：理解模型 / analyzer

1. 第一版不做大 agent，先做阶段级 analyzer。
2. 组合 VLM caption、OCR/text mask、face/person mask、saliency、edge/frequency map。
3. 输出 prompt pack、protected regions、enhance regions、risk map。
4. Controller 使用 analyzer 输出替代硬编码 attention/semantic priority。

### Phase 7：Agent loop

1. Agent 每轮只做计划，不直接生成 token。
2. 工具链：`analyze -> build_mask -> generate_candidates -> score_candidates -> commit_or_remask`。
3. 每轮输出 mode、mask ratio、strength、guidance scale、candidate K、freeze/refine regions。

### Phase 8：early commit 与 tile-global

1. 连续两轮稳定、entropy 低、LR/boundary/semantic check 通过的 token 进入 commit mask。
2. 1024 先 full image refinement；2048 以后再做 tile refinement + overlap consensus。
3. seam 或 overlap 不一致区域进入下一轮 active mask。

## 当前仓库快照

- 当前分支：`main`
- 远端状态：`main...origin/main`
- 写入本记录前工作树：干净
- 最新提交：`82ec514 Add observation consistency projection`
- 主要目录：`src/`、`agentsr/`、`tools/`、`assets/`、`train/`、`envs/`、`output/`
- 当前没有看到 `outputs/` 目录；历史实验输出只在文档中记录，当前工作树未保留这些 ignored runtime artifacts。

## 已有实现

### `agentsr/controller.py`

已实现：

- `AgentPlan`：结构化 controller plan。
- `derive_agent_plan`：从 prompt/mode/alpha 推导 mode、alpha、LR/boundary consistency weight、temperature。
- `make_outpaint_canvas`：outpaint 初始化画布和外扩 mask。
- `frequency_entropy_map`：基于梯度和局部方差的 detail map。
- `adaptive_mask`：生成 Meissonic inpaint 可用的白色重绘 / 黑色保留 mask。
- `downsample_consistency_metrics`：HR candidate downsample 到 LR 后计算 MSE/PSNR。
- `observation_consistency_project`：像素空间 LR residual projection，将 Meissonic 输出拉回 LR observation。
- `tile_grid`：目前只生成 tile metadata，还没有实际 tiled decoding。
- `build_refinement_assets`：写出 init image、mask image、plan JSON、controller metrics。

### `tools/agent_mask_sr.py`

已实现：

- CLI dry-run：生成 controller assets。
- CLI full run：加载 Meissonic `InpaintPipeline`，调用 inpaint 后输出 `meissonic_refined.png`。
- dtype 参数：默认 `float32`，用于规避历史 fp16 黑图。
- consistency projection 参数：`--skip_consistency_projection`、`--consistency_steps`、`--consistency_strength`、`--edit_strength`、`--mask_blur_radius`。
- 输出路径校验：`ensure_repo_local` 确保 CLI output_dir 在 repo 内。

### Meissonic 管线状态

- `src/pipeline_inpaint.py` 已有 image + mask inpaint 逻辑。
- inpaint 会把 mask resize 到 VQ/token 分辨率，并对 mask 区域写入 `mask_token_id`。
- `src/pipeline_img2img.py` 已有 `strength` 逻辑，强度越高越偏重新生成。
- 目前 AgentSR 侧还没有单独封装 `MeissonicTokenEditor`，也没有显式暴露 token-grid masks。

## 文档中记录的历史结果

现有 `RESULTS.md` 记录了 2026-05-01 的进展：

- controller dry-run 曾通过。
- full Meissonic 1024 fp32 pipeline 曾跑通。
- 512x512 曾在 rotary position path 报 shape mismatch。
- fp16 wrapper 曾生成全黑图，后续改为默认 float32。
- 1024 fp32 refined output 非空，但 Meissonic refinement 后 LR consistency 明显下降：PSNR 约 16.44 dB。
- `observation_consistency_project` 后 PSNR 提升到约 27.33 dB。
- 当前结论：pipeline 能跑，但仍需要把 LR consistency 从后处理投影推进到候选筛选或 token-level acceptance。

## 与新 Meissonic-SR Agent 路线的差距

- 路径配置仍指向旧目录 `AgentMaskSR`，这是下一步运行前的 blocker。
- 还没有 `meissonic_sr/` 模块化结构。
- 还没有 `MeissonicTokenEditor` 抽象。
- 还没有 VQ round-trip 测试脚本和报告。
- 还没有 img2img strength sweep baseline。
- 当前 partial mask 是 pixel mask，通过 inpaint pipeline 间接映射到 token grid；还没有显式 token mask controller。
- LR consistency 当前是 projection，不是 accept/reject、reranking 或 commit-remask。
- 还没有 candidate sampler、reranker、semantic analyzer、agent loop、early commit、tile runner。

## 最近下一步

1. 只在 `/home/ma-user/workspace/llc/AgentSR` 内修正旧路径引用。
2. 重新跑 controller dry-run，输出到 repo 内 `outputs/smoke_controller_agent_sr/`。
3. 新增 VQ round-trip 脚本和 `roundtrip_error_report.json`。
4. 新增 img2img strength sweep baseline，先测 `0.15 / 0.30 / 0.50`。
5. 将当前 pixel-space LR projection 保留为 baseline，同时实现 candidate accept/reject 评分文件。
6. 再进入 semantic analyzer 和 agent loop，不要一开始做完整自动 agent。

## 本次进展日志

- 已连接 `lsh-stable`。
- 已确认目标目录 `/home/ma-user/workspace/llc/AgentSR` 存在。
- 已确认远端仓库在 `main` 分支，最新提交为 `82ec514`。
- 已检查 `PLAN.md`、`README.md`、`RESULTS.md`、`HISTORY.md`、`CHAT.md`。
- 已检查 `agentsr/controller.py`、`tools/agent_mask_sr.py`、Meissonic inpaint/img2img 关键调用点。
- 已发现路径漂移问题：文档和环境脚本仍指向 `previous AgentMaskSR workspace path`。
- 已确认当前实现阶段：pixel-level partial mask + Meissonic inpaint + LR consistency projection。
- 已将本次检查和下一步计划记录在本文件。


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

## 2026-05-13 GitHub Push Restored

Status: completed.

- GitHub authentication was restored by using ~/.ssh/id_ed25519_github on lsh-stable.
- The AgentSR repository now has local git config core.sshCommand pointing to that key and GitHub SSH over port 443.
- origin was changed from HTTPS to git@ssh.github.com:Da1suKE66/AgentMaskSR.git.
- Checkpoint commit pushed: 57ddb3b Add token rerank SR MVP.

Policy going forward:

- Push after each meaningful implementation or experiment checkpoint.
- Keep large outputs and HF/model cache out of git; only code, configs, and markdown progress records should be committed.

## 2026-05-13 Codex CLI Install on lsh-stable

Status: completed.

- Installed Node.js under /cache/llc/codex-cli/node.
- Installed Codex CLI through npm under /cache/llc/codex-cli/npm-global.
- Verified versions: Node v22.22.2, npm 10.9.7, codex-cli 0.130.0.
- envs/agentsr_cache_env.sh now prepends the Codex CLI and Node.js bins to PATH through AGENTMASKSR_CODEX_PREFIX.

Usage:

- Run `source envs/agentsr_cache_env.sh` from AgentSR, then `codex --version`.

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

## 2026-05-13 Candidate Sweep and CLIP Metrics Checkpoint

Status: completed.

Implemented:

- Added agentsr/semantic_metrics.py with optional CLIPScorer and deterministic naturalness proxies.
- CandidateScore now records clip_text_similarity and clip_image_similarity when available.
- ScoreWeights now exposes clip_text and clip_image weights; both default to 0, so CLIP is recorded but does not affect reranking unless explicitly requested.
- tools/agent_mask_sr.py now supports --enable_clip_score, --clip_score_device, --clip_text_weight, and --clip_image_weight.
- Each round writes candidate_multimodal_metrics.json beside candidate_scores.json.
- Added tools/sweep_token_rerank.py to run reproducible parameter/candidate sweeps and aggregate run_summary.json files.
- Fixed low-strength invalid token failures: MeissonicTokenEditor now enforces at least one effective sampling step and replaces mask/out-of-range token ids with base tokens before VQ decode.

CLIP smoke:

- Output: outputs/token_rerank_uncage_k2_clip_smoke_1024
- Config: uncage, token_threshold 0.5, strength 0.25, guidance 5.0, candidate_k 2, CLIP on CPU.
- Selected candidate: seed 67.
- Candidate 0: LR L1 1.0285, CLIP text 0.2662, CLIP image 0.9906, total 5.3068.
- Candidate 1: LR L1 1.0134, CLIP text 0.2689, CLIP image 0.9919, total 5.2818.
- CLIP weights were zero, so selection remained LR/boundary driven; CLIP metrics are now available for analysis.

Parameter sweep smoke:

- Output: outputs/token_rerank_param_sweep_uncage_fixed
- Config grid: uncage, token_threshold 0.5, strength 0.10/0.15, guidance 4.0/5.0, candidate_k 2, steps 8.
- All four runs completed after the low-strength fix.
- guidance 4.0 gave candidate LR L1 0.9649 and score 4.7801.
- guidance 5.0 gave candidate LR L1 1.0158 and score 5.2198.
- All runs were correctly rejected by active_lr_worse_ratio=1.0 and stable_active_ratio=0.0.

Interpretation:

- The candidate/rerank infrastructure is now usable for controlled sweeps.
- The current dog sample still shows that Meissonic active-token edits locally worsen LR in all active tokens, even when global LR L1 stays below 1.1.
- Lower guidance is slightly safer than guidance 5.0 on this sample.
- strength 0.10 and 0.15 are equivalent with steps=8 because both result in one effective Meissonic sampling step; future sweeps should use more steps when testing strength sensitivity.

Next:

- Run broader sweeps over token_threshold 0.35/0.5, strength 0.15/0.25/0.35, guidance 3/4/5, and candidate_k 2/4.
- Add an agent planner that consumes sweep/round metrics and proposes the next mask/parameter plan.
- Start semantic acceptance checks using CLIP/image similarity first, then VLM/OCR/face analyzers when available.

## 2026-05-13 Semantic-Guided Active Mask Pilot

Status: completed.

Implemented:

- Added agentsr/semantic_guidance.py with CLIPRegionPrior.
- Added a new mask policy: semantic_uncage.
- semantic_uncage uses CLIP crop-text similarities as coarse semantic priors:
  - refine prompts increase active priority.
  - protect prompts suppress active priority.
  - the current implementation reweights the base uncage score rather than replacing it.
- Added semantic prompt CLI controls:
  - --semantic_refine_prompts
  - --semantic_protect_prompts
  - --semantic_clip_model_path
  - --semantic_clip_device
  - --semantic_grid_size
  - --semantic_batch_size
- Mask artifacts now include semantic_score_map.png and semantic_protect_map.png for semantic_uncage.

Dog 4x dry-run comparison:

- Output: outputs/semantic_mask_guidance_dog_dryrun
- Prompts:
  - refine: fine dog fur texture; animal hair detail; natural background texture
  - protect: dog eyes; dog nose; dog face outline; object boundary; text logo
- Original uncage, alpha 0.35, token_threshold 0.5:
  - active tokens: 93
  - active ratio: 0.0227
  - edge coverage top10: 0.4804
- Initial additive semantic_uncage, alpha 0.35:
  - active tokens: 435
  - active ratio: 0.1062
  - edge coverage top10: 0.2900
  - semantic_score_mean_active: 0.8754
  - semantic_protect_top25_leakage: 0.0756
- Conservative semantic_uncage, alpha 0.0:
  - active tokens: 244
  - active ratio: 0.0596
  - semantic_score_mean_active: 0.9298
  - semantic_protect_top25_leakage: 0.0121
- Reweighted semantic_uncage, alpha 0.35:
  - active tokens: 193
  - active ratio: 0.0471
  - edge coverage top10: 0.3033
  - semantic_score_mean_active: 0.6309
  - semantic_protect_top25_leakage: 0.0281

Meissonic smoke comparison:

- Baseline output: outputs/semantic_guided_active_sweep_dog/uncage_tok0p50_str0p15_gs4p0_k2
- Initial semantic output: outputs/semantic_guided_active_sweep_dog/semantic_uncage_tok0p50_str0p15_gs4p0_k2
- Reweighted semantic output: outputs/semantic_uncage_reweight_k2_smoke_1024

| policy | active tokens | candidate LR L1 | boundary L1 | CLIP text | CLIP image | accept |
|---|---:|---:|---:|---:|---:|---|
| uncage | 93 | 0.9649 | 7.2640 | 0.2666 | 0.9905 | rejected, active_lr_worse_ratio=1.0 |
| semantic_uncage additive | 435 | 4.8693 | 11.7317 | 0.2397 | 0.8472 | rejected, LR regression |
| semantic_uncage reweighted | 193 | 2.1385 | 10.5439 | 0.2756 | 0.9784 | rejected, LR regression |
| semantic_uncage reweighted, token_threshold 0.65 | 71 | 0.9765 | 9.2734 | 0.2656 | 0.9917 | rejected, active_lr_worse_ratio=1.0 |

Budget-aligned semantic check:

- Output: outputs/semantic_uncage_reweight_th0.65_k2_smoke_1024
- token_threshold 0.65 reduced semantic_uncage to 71 active tokens, closer to uncage's 93 active tokens.
- LR L1 became comparable to uncage, but boundary L1 stayed worse: 9.2734 vs 7.2640.
- CLIP image similarity improved slightly: 0.9917 vs 0.9905.
- CLIP text similarity did not improve: 0.2656 vs 0.2666.
- This means the current CLIP prior can steer active region selection, but it is not yet a quality win.

Interpretation:

- CLIP semantic prior does guide active regions: it lowers structural edge coverage and raises semantic/refine-region priority.
- The first naive semantic mask does not improve SR consistency on the dog test. It increases active area and worsens LR/boundary consistency.
- Reweighting semantic prior through uncage is better than additive replacement, but still worse than the original smaller uncage mask on LR consistency.
- The useful role for CLIP/VLM is currently protection/reranking/planning, not direct mask expansion.
- Next semantic mask version should keep the active token budget matched to baseline and use semantic prior mainly to suppress protected regions or choose among same-budget candidate tokens.

Next:

- Add a same-budget semantic token selector: start from uncage token candidates, then swap tokens by semantic refine/protect score without increasing active token count.
- Add semantic protect hard-freeze masks for eyes/nose/text/object boundary before active token selection.
- Evaluate semantic guidance with matched active token counts, not only matched pixel mask budget.

## 2026-05-13 Final Output Semantics Fix

Status: completed.

Problem:

- token_rerank_sr wrote rejected candidates to round_*/candidate_*.png but skipped updating current_image before the next loop.
- As a result, final_hr.png could equal x_base_hr.png whenever the best generated candidate failed the strict accept/commit gate.
- That is wrong for the main method: x_base_hr is a deterministic baseline/ablation, not the main final result.

Fix:

- Rejection now means "do not commit these tokens", not "rollback final output to bicubic".
- The best generated candidate is always used as the current SR image and can become final_hr.png.
- If a candidate is rejected, proposed new commits are not kept; those token positions remain active/remasked for the next round.
- run_summary.json now records:
  - final_source
  - used_as_current
  - committed

Verification run:

- Output: outputs/semantic_uncage_reweight_th0.65_k2_smoke_1024_finalfix
- final_source: best_generated_candidate
- x_base_hr hash: e57e58d9fa58
- final_hr hash: e672fef431d1
- final_hr equals selected candidate_01.
- final vs base mean absolute RGB difference: [0.5030, 0.4586, 0.4370]
- The selected candidate is still marked rejected for consistency:
  - reject_reason: active_lr_worse_ratio:1.000000>0.500000
  - committed: false
  - active tokens remain 71 for remasking.

Interpretation:

- The pipeline now separates generation output from consistency acceptance.
- final_hr.png is the Meissonic-generated SR branch.
- x_base_hr.png remains available only as baseline/ablation and as the deterministic starting observation.

## 2026-05-13 Progressive 2x Token-Rerank Pipeline

Status: completed.

Rationale:

- The previous token_rerank_sr path did LR -> one-shot bicubic 1024 -> Meissonic local detail refinement.
- That made Meissonic behave like a detail enhancer on a very blurry 1024 base.
- The main path should instead refine progressively:
  - 256 -> 512 with Meissonic token refinement
  - 512 -> 1024 with Meissonic token refinement
- The old single-stage token_rerank_sr remains available as an ablation.

Implemented:

- Added CLI modes:
  - progressive_token_rerank_sr
  - cascaded_token_rerank_sr
- Added --progressive_scale_factor, default 2.
- Progressive mode creates stage directories:
  - stage_01_512x512
  - stage_02_1024x1024
- Each stage writes:
  - stage_input.png
  - x_base_stage.png
  - mask artifacts
  - round_*/candidate_*.png
  - stage_final.png
- Scoring is now stage-aware:
  - candidate_stage_ref_l1 compares the candidate downsampled to the previous stage image.
  - candidate_original_lr_l1 also tracks consistency to the original LR image.
- final_hr.png comes from the last stage generated candidate branch.
- tools/sweep_token_rerank.py now accepts --mode so sweeps can run progressive_token_rerank_sr.

Meissonic non-1024 fix:

- The first real 512 smoke exposed a rotary positional id mismatch in src/pipeline_inpaint.py.
- The old non-1024 branch doubled the latent grid before calling _prepare_latent_image_ids.
- For a 512 input, transformer hidden tokens after down_block require 333 rotary ids, but the old branch produced 1101.
- Fixed by using the actual latent grid for all inpaint sizes.

Verification:

- Dry-run output: outputs/progressive_uncage_dog_dryrun
- Real smoke output: outputs/progressive_uncage_dog_k1_smoke_fixed
- Stage sizes: 256 -> 512 -> 1024.
- Stage 1 token grid: 32x32, active tokens 46.
- Stage 2 token grid: 64x64, active tokens 18 after stage 1 generated input.
- final_source: best_generated_candidate.
- final_hr.png is generated by the progressive branch.

Current quality result:

- The pipeline now runs, but quality is not solved.
- Stage 1 candidate regressed stage-reference L1 from 0.6146 to 2.1129 and was not committed.
- Stage 2 candidate slightly improved original LR L1 from 2.1356 to 2.0922 but still failed active-stage consistency.
- This confirms the next work should be stage-specific mask/threshold/strength tuning, not returning to one-shot 1024.

Next:

- Use smaller stage-1 active budgets or stronger protected rings to prevent early-stage drift.
- Sweep progressive mode over token_threshold 0.65/0.75, strength 0.10/0.15, guidance 3/4.
- Add per-stage defaults: stage 1 conservative structure recovery, stage 2 texture refinement.

## 2026-05-14 Meissonic-Aware Progressive Sweep

Status: completed.

Implemented:

- Added stage-specific progressive controls:
  - --stage_token_mask_thresholds
  - --stage_refine_strengths
  - --stage_guidance_scales
  - --stage_steps
  - --temperature_start
  - --temperature_end
- Added tools/summarize_progressive_experiments.py.
- Generated:
  - outputs/progressive_experiment_summary.json
  - outputs/progressive_experiment_summary.md
- Updated progressive defaults from the best current dog run:
  - thresholds: 0.70,0.60
  - strengths: 0.18,0.16
  - guidance: 3.0,4.0
  - steps: 24,24
  - temperature: 0.5 -> 0.0
  - rounds default: 1

Best result:

- Run: outputs/progressive_exp_stage1_ultra_stage2_conservative_k2
- Active tokens: stage 1 = 6, stage 2 = 11.
- Final PSNR downsampled to LR: 37.3643.
- This improves over the first progressive smoke: 28.3105.

Key findings:

- Stage 1 must be extremely conservative; reducing stage 1 active tokens from 46 to 6 is the largest improvement.
- K=2 candidate reranking helps.
- 48 steps and temperature 2.0, closer to the Meissonic paper's default sampling style, did not improve this SR setup.
- Two rounds currently worsen results; commit/remask still needs a stronger acceptance policy.
- semantic_uncage still expands active masks too much and should become same-budget semantic token selection.
