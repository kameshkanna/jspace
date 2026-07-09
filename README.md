# jspace — Jacobian Lens & Global Workspace Replication

Open replication of the J-lens / Global Workspace paper
["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, 2026) — rebuilt from scratch for open HuggingFace models.

---

## What this does

| Component | What it computes |
|---|---|
| `JacobianLens` | Averaged Jacobian readout vectors across a prompt corpus — "what is each position poised to say?" |
| `WorkspaceAnalyzer` | Per-layer active concept count, entropy, variance explained, phase detection (Early / Workspace / Output) |
| `viz.py` | Layer profile, concept heatmap, capacity comparison plots |

---

## Setup (Lambda Labs H100)

```bash
git clone <this-repo>
cd jspace
bash setup.sh                                      # Qwen 7B (default)
bash setup.sh meta-llama/Meta-Llama-3.1-8B-Instruct   # or Llama 3.1 8B
```

---

## Quick start

```bash
source .venv/bin/activate

# Step 1 — fit J-lens (once per model, ~5–15 min on H100)
python scripts/compute_jlens.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output outputs/jlens_qwen7b.pt

python scripts/compute_jlens.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output outputs/jlens_llama3_8b.pt

# Step 2 — run workspace analysis
python scripts/run_workspace.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --jlens outputs/jlens_qwen7b.pt \
    --prompts_file configs/lang_pivot_prompts.txt \
    --save_dir outputs/figs_qwen7b

python scripts/run_workspace.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --jlens outputs/jlens_llama3_8b.pt \
    --prompts_file configs/lang_pivot_prompts.txt \
    --save_dir outputs/figs_llama3_8b
```

Figures are saved under `outputs/figs_*/`.

---

## Supported models

Any HuggingFace causal LM with the standard `model.layers` structure:
- Qwen2 / Qwen2.5 family (default: 7B-Instruct)
- LLaMA 2 / 3 / 3.1 (comparison: Meta-Llama-3.1-8B-Instruct)
- Mistral / Mixtral
- Gemma / Gemma2

For GPT-2 / GPT-J / Falcon / OPT the layer path is auto-detected.

---

## Language pivot experiment

The key experiment: does Qwen internally process in Chinese regardless of input language?

```bash
# Fit J-lens for both models
python scripts/compute_jlens.py --model Qwen/Qwen2.5-7B-Instruct   --output outputs/jlens_qwen7b.pt
python scripts/compute_jlens.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --output outputs/jlens_llama3_8b.pt

# Run the same multilingual prompts on both
python scripts/run_workspace.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --jlens outputs/jlens_qwen7b.pt \
    --prompts_file configs/lang_pivot_prompts.txt \
    --save_dir outputs/figs_qwen7b

python scripts/run_workspace.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --jlens outputs/jlens_llama3_8b.pt \
    --prompts_file configs/lang_pivot_prompts.txt \
    --save_dir outputs/figs_llama3_8b
```

Expected: Qwen middle layers show Chinese tokens on English input. Llama does not.

---

## Python API

```python
from jspace import HookedModel, JacobianLens, WorkspaceAnalyzer

model = HookedModel("Qwen/Qwen2.5-7B-Instruct")

jlens = JacobianLens(model, use_full_jacobian=False)
jlens.fit(prompts, max_length=128)
jlens.save("outputs/jlens_qwen7b.pt")

analyzer = WorkspaceAnalyzer(jlens)  # entropy threshold auto-scaled to vocab
report = analyzer.analyse("A spider has 8 legs, so three spiders have")

WorkspaceAnalyzer.print_report(report)
print(f"Workspace layers: {report.workspace_start}–{report.workspace_end}")
print(f"Peak capacity: {report.peak_capacity} active positions")
```

---

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--full_jacobian` | off | Full vocab Jacobian — accurate, needs ~40GB+ VRAM |
| `--n_prompts` | 100 | More = better averaging, more time |
| `--entropy_threshold` | auto | Auto-scaled to 60% of max vocab entropy; override manually if needed |
| `--var_threshold` | 0.05 | Min J-space variance for workspace label |

---

## Connection to ManifoldSteer

The paper finds **10–25 active concepts** in the workspace at any layer.
ManifoldSteer found ~5–6 intrinsic dimensions per concept →
geometric capacity ≈ **512 simultaneous concepts**.

This repo measures the actual throughput (10–25) vs the geometric
capacity (512) — the bottleneck between geometric headroom and observed utilization.

---

## Estimated H100 cost

| Model | Step | Time | VRAM |
|---|---|---|---|
| Qwen2.5-7B | `compute_jlens.py` (approx Jacobian, 35 prompts) | ~5–10 min | ~16 GB |
| Llama-3.1-8B | `compute_jlens.py` (approx Jacobian, 35 prompts) | ~5–10 min | ~16 GB |
| Either | `run_workspace.py` per prompt | ~45 sec | ~16 GB |
