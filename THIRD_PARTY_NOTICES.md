# Third-party notices

jspace is MIT-licensed (see `LICENSE`). Parts of it are adapted from the projects below.

## J++ Lens — safety-research/jpp_lens

Copyright 2026 Kola Ayonrinde. Licensed under the Apache License, Version 2.0
(`third_party/jpp_lens-LICENSE-Apache-2.0.txt`). https://github.com/safety-research/jpp_lens

Ayonrinde & Lindsey (2026), *J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses*.

Adapted, with changes, in this repository:

- `src/jspace/lrp.py` — the LN, identity and half LRP rules (reimplemented for the dense
  Qwen2/Qwen3/Llama/Mistral classes only; no MoE or multi-stream rules).
- `src/jspace/routing.py` — the normalise → centre → PCA → k-means++ activation router
  (reimplemented; same pipeline).
- `src/jspace/readout_filter.py` — the Readout Filtering rule (tokens with no letter or digit).
- `src/jspace/expert_weights.py` — the expert-weight search: linearised `unembed · rms`
  logit cache, sigmoid-margin surrogate of recall@k, multi-start Adam keeping the best exact
  iterate (reimplemented, extended to per-offset weights and to future-token targets).

The LRP rules in jpp_lens derive from the R-Lens implementation of Camila Blank and Agam
Bhatia (https://huggingface.co/camilablank/workspace-lenses).

## Jacobian lens — anthropics/jacobian-lens

Copyright 2026 Anthropic PBC, Apache License 2.0. https://github.com/anthropics/jacobian-lens

The latent-variable readout items used by `scripts/eval_offset_lenses.py` are read from a
local copy of that repository's `data/jlens/evaluations/` (not redistributed here).
