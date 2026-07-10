# Contributing to jspace

Thanks for your interest. This repo is an open replication of the Jacobian Lens / Global
Workspace paper (Lindsey et al., Anthropic 2026). Contributions that improve correctness,
extend coverage to new models, or add new experiments are welcome.

---

## How the repo works

```
jspace/
├── src/jspace/
│   ├── model.py        # HookedModel — loads any HF causal LM, exposes residual-stream hooks
│   ├── jlens.py        # JacobianLens — fused Hutchinson VJP estimator + corpus J-lens vectors
│   ├── workspace.py    # WorkspaceAnalyzer — 4-signal majority-vote workspace detection
│   └── viz.py          # Plotting: layer profile, concept heatmap, capacity comparison
├── scripts/
│   ├── compute_jlens.py    # CLI: fit J-lens matrices over a prompt corpus, save .pt
│   └── run_workspace.py    # CLI: analyse prompts using a fitted J-lens, save figures
├── configs/
│   ├── prompts.txt              # 1000-prompt corpus for J-lens fitting
│   └── lang_pivot_prompts.txt   # 14-prompt language pivot experiment (EN/ZH/JA)
└── results_qwen7b_vs_llama3_8b.txt   # Full results: methodology, findings, model comparison
```

### Two-step workflow

**Step 1 — fit J-lens** (`compute_jlens.py`): Runs one forward pass + `n_proj` backward
passes per prompt over the 1000-prompt corpus. For each layer l, accumulates the
averaged Jacobian matrix `J_l = E[dh_final / dh_l]` via the Hutchinson VJP estimator
(Rademacher random projections). Also builds `corpus_jh`: unit-normed `J_l @ h` vectors
for every (prompt, layer) pair — used downstream for gradient pursuit.
Output: a `.pt` file containing `avg_jacobians` and `corpus_jh` for all layers.

**Step 2 — workspace analysis** (`run_workspace.py`): For each prompt, runs one forward
pass and computes four signals per layer using the fitted J-lens:
1. `n_active` — nonneg gradient pursuit over `corpus_jh`; minimum k vectors needed to
   explain 95% of `J_l @ h` (paper's active concept count)
2. `next_token_acc` — fraction of positions where `argmax(lens(h_l_t)) == token_{t+1}`
3. `pos_autocorr` — mean cosine similarity of adjacent readout distributions
4. entropy valley — mean entropy in hunchback dip

Layers scoring ≥3/4 are Workspace candidates; the largest contiguous run is kept.
Output: rich table in terminal + PNG figures per prompt.

---

## Paper formulas

```
J_l       = E_{t, t'>=t, prompt} [ dh_final_{t'} / dh_l_t ]
lens(h_l) = softmax(W_U · norm(J_l @ h_l))
```

The readout `lens(h_l)` maps a residual-stream hidden state at layer l to a vocabulary
distribution — showing which tokens the model "has in mind" at that layer and position.

---

## Getting started

```bash
git clone https://github.com/kameshkanna/jspace
cd jspace
bash setup.sh
source .venv/bin/activate

# quick sanity check (3 min on H100)
python scripts/compute_jlens.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --n_prompts 100 --n_proj 16 \
    --output outputs/jlens_test.pt

python scripts/run_workspace.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --jlens outputs/jlens_test.pt \
    --prompt "The capital of France is"
```

---

## Types of contributions

**Bug reports** — open an issue using the Bug Report template. Include model, n_prompts,
n_proj, the exact error, and steps to reproduce.

**New model support** — if `HookedModel._discover_layers()` fails for your model, add the
layer path to the candidates list in [model.py](src/jspace/model.py#L69). Open a PR with
the model name and a quick test showing workspace detection works.

**New experiments** — new prompt sets, language pivot on more languages, scaling experiments
across model families. Add a prompt file under `configs/` and results under a new
`results_*.txt` at the repo root.

**Paper fidelity fixes** — if you find a gap between this implementation and the paper
(Lindsey et al. 2026), open an issue with the specific equation and what differs.

---

## Pull request checklist

- [ ] Imports are absolute (compatible with `src/` layout)
- [ ] All functions and methods have type hints
- [ ] No bare `except` or `except Exception`
- [ ] No Python loops over tensors — use vectorized ops
- [ ] New logic has a unit test or a comment explaining why testing is impractical
- [ ] `python scripts/run_workspace.py --prompt "The capital of France is"` still works

---

## Commit style

```
type(scope): short description

# types: feat, fix, perf, refactor, docs, chore
# examples:
feat(workspace): add head-level per-head J-lens decomposition
fix(jlens): handle seq_len=1 edge case in fused estimator
docs: add example for Mistral 7B
```

No ticket IDs needed. Keep the subject line under 72 characters.
