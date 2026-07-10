#!/usr/bin/env bash
# Full paper-faithful replication run — Qwen2.5-7B + Llama-3.1-8B
# Hardware: any single 80GB GPU
#
# Usage:
#   bash run_experiments.sh          # runs both models end to end
#   bash run_experiments.sh qwen     # Qwen only
#   bash run_experiments.sh llama    # Llama only
set -euo pipefail

TARGET="${1:-both}"

git pull
source .venv/bin/activate

mkdir -p outputs/qwen7b outputs/llama3_8b

# ─────────────────────────────────────────────────────────────────────────────
# QWEN 2.5 7B
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$TARGET" == "both" || "$TARGET" == "qwen" ]]; then
    echo "============================================================"
    echo "  QWEN2.5-7B — Step 1: fit J-lens (1000 prompts, 32 proj)"
    echo "  Step 1: fit J-lens"
    echo "============================================================"
    python scripts/compute_jlens.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --prompts_file configs/prompts.txt \
        --n_prompts 1000 \
        --n_proj 32 \
        --output outputs/qwen7b/jlens.pt

    echo ""
    echo "============================================================"
    echo "  QWEN2.5-7B — Step 2: workspace analysis (14 lang-pivot prompts)"
    echo "  Step 2: workspace analysis"
    echo "============================================================"
    python scripts/run_workspace.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --jlens outputs/qwen7b/jlens.pt \
        --prompts_file configs/lang_pivot_prompts.txt \
        --save_dir outputs/qwen7b/figs
fi

# ─────────────────────────────────────────────────────────────────────────────
# LLAMA 3.1 8B
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$TARGET" == "both" || "$TARGET" == "llama" ]]; then
    echo ""
    echo "============================================================"
    echo "  LLAMA-3.1-8B — Step 1: fit J-lens (1000 prompts, 32 proj)"
    echo "  Step 1: fit J-lens"
    echo "============================================================"
    python scripts/compute_jlens.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --prompts_file configs/prompts.txt \
        --n_prompts 1000 \
        --n_proj 32 \
        --output outputs/llama3_8b/jlens.pt

    echo ""
    echo "============================================================"
    echo "  LLAMA-3.1-8B — Step 2: workspace analysis (14 lang-pivot prompts)"
    echo "  Step 2: workspace analysis"
    echo "============================================================"
    python scripts/run_workspace.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --jlens outputs/llama3_8b/jlens.pt \
        --prompts_file configs/lang_pivot_prompts.txt \
        --save_dir outputs/llama3_8b/figs
fi

echo ""
echo "============================================================"
echo "  DONE — outputs:"
echo "    outputs/qwen7b/figs/       — PNG figures"
echo "    outputs/llama3_8b/figs/    — PNG figures"
echo "    outputs/qwen7b/jlens.pt    — fitted J-lens (reusable)"
echo "    outputs/llama3_8b/jlens.pt — fitted J-lens (reusable)"
echo "============================================================"
