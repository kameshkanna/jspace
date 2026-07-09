# jspace — Jacobian Lens & Global Workspace Replication

Open replication of the J-lens / Global Workspace paper
["Verbalizable Representations Form a Global Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)
(Lindsey et al., Anthropic 2026) — rebuilt from scratch for open HuggingFace models.

---

## What this does

| Component | What it computes |
| --- | --- |
| `JacobianLens` | Full (d_model × d_model) averaged Jacobian matrices via Hutchinson VJP estimator; builds corpus J-lens vectors for gradient pursuit |
| `WorkspaceAnalyzer` | Per-layer active concept count via gradient pursuit (min-k matching pursuit, 95% coverage), entropy, variance, phase detection |
| `viz.py` | Layer profile, concept heatmap, capacity comparison plots |

**Paper formula implemented:**

```text
J_l = E_{t, t'>=t, prompt} [ dh_final_{t'} / dh_l_t ]
lens(h_l) = softmax(W_U · norm(J_l @ h_l))
```

---

## Setup (Lambda Labs H100 80GB)

```bash
git clone <this-repo>
cd jspace
bash setup.sh                                           # Qwen 7B (default)
bash setup.sh meta-llama/Meta-Llama-3.1-8B-Instruct    # or Llama 3.1 8B
```

---

## Quick start

```bash
source .venv/bin/activate

# Step 1 — fit J-lens (once per model)
# H100 80GB: ~15 min for 100 prompts x 32 layers x 16 projections
python scripts/compute_jlens.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n_prompts 100 --n_proj 16 \
    --output outputs/jlens_qwen7b.pt

python scripts/compute_jlens.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --n_prompts 100 --n_proj 16 \
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

---

## Python API

```python
from jspace import HookedModel, JacobianLens, WorkspaceAnalyzer

model = HookedModel("Qwen/Qwen2.5-7B-Instruct")

jlens = JacobianLens(model, n_proj=16)
jlens.fit(prompts, max_length=128)
jlens.save("outputs/jlens_qwen7b.pt")

analyzer = WorkspaceAnalyzer(jlens)
report = analyzer.analyse("A spider has 8 legs, so three spiders have")

WorkspaceAnalyzer.print_report(report)
print(f"Workspace layers: {report.workspace_start}–{report.workspace_end}")
print(f"Peak capacity: {report.peak_capacity} active positions")
```

---

## Key flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--n_proj` | 16 | Hutchinson projections; 16=fast, 32=paper quality |
| `--n_prompts` | 100 | More = better J averaging (paper uses 1000) |
| `--var_threshold` | 0.05 | Min J-space variance fraction for workspace label |

---

## Supported models

Any HuggingFace causal LM with the standard `model.layers` structure:

- Qwen2 / Qwen2.5 family (default: 7B-Instruct)
- LLaMA / Llama 3 / 3.1
- Mistral / Mixtral
- Gemma / Gemma2

GPT-2 / GPT-J / Falcon / OPT layer paths are auto-detected.

---

## Language pivot experiment

Tests whether Qwen internally processes in Chinese regardless of input language.

```bash
# Same 14 prompts (English, Chinese, Japanese) on both models
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

## Connection to ManifoldSteer

The paper finds **10–25 active concepts** in the workspace at any layer.
ManifoldSteer found ~5–6 intrinsic dimensions per concept →
geometric capacity ≈ **512 simultaneous concepts**.

This repo measures the actual throughput (10–25) vs the geometric
capacity (512) — the bottleneck between geometric headroom and observed utilization.

---

## H100 80GB cost estimates

| Model | n_prompts | n_proj | Time | VRAM |
| --- | --- | --- | --- | --- |
| 7–8B | 100 | 16 | ~15 min | ~20 GB |
| 7–8B | 100 | 32 | ~30 min | ~20 GB |
| 7–8B | 1000 | 16 | ~2.5 hrs | ~20 GB |
| 7–8B per-prompt workspace run | — | — | ~45 sec | ~16 GB |
