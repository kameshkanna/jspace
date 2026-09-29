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

## Multi-token J-space (offset-resolved lens)

An extension of the J-lens above.  The paper's `J_l` averages the Jacobian over every future
target position, which collapses *when* a residual-stream feature is used.  The offset lens
keeps that time axis:

```
J_l^(Δ)       = E_{prompt, t} [ ∂h_final_{t+Δ} / ∂h_l_t ]        Δ ∈ {0, 1, 2, 3, 4, 6, 8}
lens_Δ(h_l_t) = softmax(W_U · norm(J_l^(Δ) h_l_t))              → reads token t+Δ+1
```

| Component | What it computes |
| --- | --- |
| `OffsetJacobianLens` | `J_l^(Δ)` for every (layer, Δ) from the same backward passes: each target position gets its own Rademacher probe, comb-masked so the dominant Δ=0 term cannot leak into Δ>0. Projections are split into two independent halves, giving an unbiased temporal reach `‖J_l^(Δ)‖` and a split-half reliability score per cell. |
| `MultiTokenWorkspace` | Latent strings (one readout per Δ at every position), lookahead accuracy against the actual token Δ+1 ahead with a repetition baseline, and joint vs per-offset concept counts over mean-centred corpus atoms. |
| `plot_temporal_reach`, `plot_lookahead_profile` | Reach + reliability heatmaps; per-prompt lookahead and concept-count profiles. |

```bash
# fit (packs the corpus into 128-token sequences; accumulators need 2 × layers × K × d² × 4 B)
python scripts/fit_offset_jlens.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layers 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
    --n_proj 64 \
    --output outputs/qwen7b/offset_jlens.pt

# analyse
python scripts/run_multitoken.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --lens outputs/qwen7b/offset_jlens.pt \
    --prompts_file configs/lang_pivot_prompts.txt \
    --save_dir outputs/qwen7b/multitoken
```

**Validation.** `tests/test_offset_lens.py` checks the estimator against exact autograd Jacobians
on a random Llama: error falls as 1/√n_proj (unbiased), the split-half reach is within 8% of the
true norm, and the final-layer structure is recovered exactly.  `pytest` runs on CPU in ~10 s.

**First result (Qwen2.5-0.5B-Instruct, 1000 prompts, n_proj=32).** Relative temporal reach is
highest in layers 0–13 and falls sharply after layer 14.  Δ>0 latent strings do **not**
predict future tokens (lookahead accuracy 0.00 at every layer), and this also holds for the
exact per-instance contribution `J(t,t+Δ)·h_t` computed with forward-mode autodiff — so it is a
property of what a position sends forward in this model, not estimator noise.  Treat Δ>0
latent strings as unvalidated until a model shows lookahead accuracy above the reported
baseline.  With 1000 last-token atoms, concept pursuit rarely reaches 95% coverage (see the
`conv` column), so concept counts are dictionary-limited at this corpus size.

---

## Paper

Lindsey et al., "Verbalizable Representations Form a Global Workspace in Language Models", Anthropic 2026.
https://transformer-circuits.pub/2026/workspace/index.html
