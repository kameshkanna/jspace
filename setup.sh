#!/usr/bin/env bash
# Lambda Labs H100 setup — run once after cloning
# Usage: bash setup.sh [MODEL_ID]
# Default model: Qwen/Qwen2.5-7B-Instruct
# Other targets: meta-llama/Meta-Llama-3.1-8B-Instruct
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-7B-Instruct}"
VENV_DIR=".venv"

echo "=== jspace setup on Lambda Labs H100 ==="

# ── 1. Python venv ──────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet

# ── 2. Install project (editable) ──────────────────────────────────────────
pip install -e ".[dev]" --quiet

# ── 3. Verify CUDA ─────────────────────────────────────────────────────────
python3 - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("WARNING: CUDA not available — will run on CPU (slow)")
else:
    dev = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"CUDA OK — {dev} ({mem:.0f} GB)")
    print(f"PyTorch {torch.__version__}")
EOF

# ── 4. Pre-download model weights ──────────────────────────────────────────
echo ""
echo "Downloading model: $MODEL_ID"
python3 - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
model_id = "$MODEL_ID"
print(f"  tokenizer ...")
AutoTokenizer.from_pretrained(model_id)
print(f"  weights   ...")
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
print(f"  done — cached at {os.path.expanduser('~/.cache/huggingface')}")
EOF

echo ""
echo "=== Setup complete ==="
echo "Activate venv:   source $VENV_DIR/bin/activate"
echo ""
echo "Targets supported:"
echo "  Qwen/Qwen2.5-7B-Instruct          (default)"
echo "  meta-llama/Meta-Llama-3.1-8B-Instruct"
echo ""
echo "Compute J-lens:  python scripts/compute_jlens.py --model $MODEL_ID"
echo "Run workspace:   python scripts/run_workspace.py --model $MODEL_ID"
