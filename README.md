# jspace — Jacobian Lens & Global Workspace Replication

Open replication of **"Verbalizable Representations Form a Global Workspace in Language Models"**
([Lindsey et al., Anthropic 2026](https://transformer-circuits.pub/2026/workspace/index.html))
— rebuilt from scratch for open HuggingFace models.

The paper shows transformer LMs form a **global workspace** in their middle layers: ~10–25
simultaneously active concepts that are verbally interpretable and broadcast across the network.
Models also think in their dominant pretraining language internally regardless of input language
(Qwen in Chinese, Llama in English).

This repo replicates both findings on Qwen2.5-7B and Llama-3.1-8B using the paper's exact
Jacobian Lens formulation, Hutchinson VJP estimator, and 4-signal majority-vote workspace detector.

---

## What this does

| Component | What it computes |
| --- | --- |
| `JacobianLens` | Full (d_model × d_model) averaged Jacobian matrices via Hutchinson VJP estimator; corpus J-lens vectors for gradient pursuit |
| `WorkspaceAnalyzer` | Four paper signals: n_active (gradient pursuit), next_token_acc, pos_autocorr, entropy valley; 4-signal majority-vote phase detection |
| `viz.py` | Layer profile, concept heatmap, capacity comparison plots |

**Paper formulas implemented:**

```
J_l       = E_{t, t'>=t, prompt} [ dh_final_{t'} / dh_l_t ]
lens(h_l) = softmax(W_U · norm(J_l @ h_l))

Workspace signals (4-signal majority vote, >=3 required):
  1. n_active      — nonneg gradient pursuit over corpus J-lens vectors, 95% coverage
  2. next_token_acc — argmax(lens(h_l_t)) == token_{t+1} accuracy
  3. pos_autocorr  — mean cosine similarity of adjacent readout distributions
  4. entropy valley — mean_entropy in hunchback dip
```

---

## Setup

```bash
git clone https://github.com/kameshkanna/jspace
cd jspace
bash setup.sh                                           # Qwen 7B (default)
bash setup.sh meta-llama/Meta-Llama-3.1-8B-Instruct    # or Llama 3.1 8B
```

---

## Quick start

```bash
source .venv/bin/activate

# Step 1 — fit J-lens (once per model)
python scripts/compute_jlens.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output outputs/jlens_qwen7b.pt

# Use --n_prompts 100 --n_proj 16 for a quick check
python scripts/compute_jlens.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n_prompts 100 --n_proj 16 \
    --output outputs/jlens_qwen7b_fast.pt

# Step 2 — run workspace analysis
python scripts/run_workspace.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --jlens outputs/jlens_qwen7b.pt \
    --prompts_file configs/lang_pivot_prompts.txt \
    --save_dir outputs/figs_qwen7b
```

---

## Python API

```python
from jspace import HookedModel, JacobianLens, WorkspaceAnalyzer

model = HookedModel("Qwen/Qwen2.5-7B-Instruct")

jlens = JacobianLens(model, n_proj=32)
jlens.fit(prompts, max_length=128)
jlens.save("outputs/jlens_qwen7b.pt")

analyzer = WorkspaceAnalyzer(jlens)
report = analyzer.analyse("A spider has 8 legs, so three spiders have")

WorkspaceAnalyzer.print_report(report)
print(f"Workspace: L{report.workspace_start}–{report.workspace_end}")
print(f"Peak capacity: {report.peak_capacity} active concepts")
```

---

## Key flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--n_proj` | 32 | Hutchinson projections; 32 = paper quality, 16 = fast |
| `--n_prompts` | 1000 | Corpus size; paper uses 1000 |

---

## Supported models

Any HuggingFace causal LM with the standard `model.layers` structure:

- Qwen2 / Qwen2.5 family (default: 7B-Instruct)
- LLaMA / Llama 3 / 3.1
- Mistral / Mixtral
- Gemma / Gemma2
- GPT-2 / GPT-J / Falcon / OPT (auto-detected)

---

## Language pivot experiment

14 prompts covering English factual, English reasoning, emotional/social, Chinese, and Japanese.
Run both models and compare workspace concepts layer by layer.

```bash
bash run_experiments.sh          # both models end to end (~50 min on H100)
bash run_experiments.sh qwen     # Qwen only
bash run_experiments.sh llama    # Llama only
```

**Results (1000 prompts, n_proj=32):**

| Model | EN input → workspace concept | ZH input → workspace concept | Workspace depth |
| --- | --- | --- | --- |
| Qwen2.5-7B | Chinese tokens (时间和, 总共, 距离) | Chinese tokens | L22–24 / ~79% |
| Llama-3.1-8B | English tokens (physics, distances) | English tokens (physics, how) | L19–26 / ~65% |

Both models think in their dominant pretraining language internally, regardless of input language.
Full results and methodology: [results_qwen7b_vs_llama3_8b.txt](results_qwen7b_vs_llama3_8b.txt)

---

## Paper

Lindsey et al., "Verbalizable Representations Form a Global Workspace in Language Models", Anthropic 2026.
https://transformer-circuits.pub/2026/workspace/index.html
