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
