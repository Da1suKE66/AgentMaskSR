#!/usr/bin/env bash
set -euo pipefail

export AGENTMASKSR_ROOT="/home/ma-user/workspace/llc/AgentSR"
export AGENTMASKSR_CONDA_PREFIX="/cache/llc/SR"
export CONDA_PKGS_DIRS="/cache/llc/SR-pkgs"
export PIP_CACHE_DIR="/cache/llc/SR-pip-cache"
export HF_HOME="/cache/llc/SR-hf-cache"
export TRANSFORMERS_CACHE="/cache/llc/SR-hf-cache/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export MPLCONFIGDIR="/cache/llc/SR-mplconfig"
export AGENTMASKSR_CODEX_PREFIX="/cache/llc/codex-cli"
export PATH="${AGENTMASKSR_CODEX_PREFIX}/npm-global/bin:${AGENTMASKSR_CODEX_PREFIX}/node/bin:${PATH}"
export PYTHONNOUSERSITE=1

cd "${AGENTMASKSR_ROOT}"
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate "${AGENTMASKSR_CONDA_PREFIX}"
