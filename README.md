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
bash setup.sh                          # installs deps, downloads Qwen2.5-3B-Instruct
```

Default model: `Qwen/Qwen2.5-3B-Instruct`. Pass a different model:

```bash
bash setup.sh meta-llama/Meta-Llama-3-8B-Instruct
```

---

## Quick start

```bash
source .venv/bin/activate

# Step 1 — fit J-lens (run once, ~5–10 min on H100 for 100 prompts)
python scripts/compute_jlens.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --n_prompts 100 \
    --output outputs/jlens.pt

# Step 2 — analyse a prompt
python scripts/run_workspace.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --jlens outputs/jlens.pt \
    --prompt "A spider has 8 legs, so three spiders have" \
    --save_dir outputs/figs
```

Figures are saved under `outputs/figs/`.

---

## Python API

```python
from jspace import HookedModel, JacobianLens, WorkspaceAnalyzer

model = HookedModel("Qwen/Qwen2.5-3B-Instruct")

jlens = JacobianLens(model, use_full_jacobian=False)
jlens.fit(prompts, max_length=128)
jlens.save("outputs/jlens.pt")

analyzer = WorkspaceAnalyzer(jlens, entropy_threshold=3.0)
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
| `--entropy_threshold` | 3.0 | Lower = stricter active-concept criterion |
| `--var_threshold` | 0.05 | Min J-space variance for workspace label |

---

## Connection to ManifoldSteer

The paper finds **10–25 active concepts** in the workspace at any layer.
Your ManifoldSteer project found ~5–6 intrinsic dimensions per concept →
geometric capacity ≈ **512 simultaneous concepts**.

This repo lets you measure the actual throughput (10–25) vs the geometric
capacity (512) and ask: **what is the bottleneck mechanism?**

---

## Supported models

Any HuggingFace causal LM with the standard `model.layers` structure:
- Qwen2 / Qwen2.5 family (default)
- LLaMA 2 / 3
- Mistral / Mixtral
- Gemma / Gemma2

For GPT-2/GPT-J/Falcon/OPT the layer path is auto-detected.

---

## Estimated H100 cost

| Step | Time | VRAM |
|---|---|---|
| `compute_jlens.py` (100 prompts, approx Jacobian) | ~5–10 min | ~10 GB |
| `compute_jlens.py` (100 prompts, full Jacobian) | ~60–90 min | ~40 GB |
| `run_workspace.py` per prompt | ~30 sec | ~10 GB |
