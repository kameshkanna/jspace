"""
Global Workspace analysis — paper-faithful implementation.

Paper (Lindsey et al., 2026) workspace detection uses FOUR signals per layer:

  1. n_active  (sparse nonneg gradient pursuit — paper exact)
     Minimum k corpus J-lens vectors, with nonneg coefficients, whose linear
     combination explains >=95% of J_l @ h_query norm-squared.
     Each step: pick corpus vector with highest POSITIVE dot to residual,
     subtract with a = max(dot/||c||^2, 0).  Stops when max positive dot = 0.
     Averaged across sequence positions.  Falls back to excess-kurtosis when
     corpus_jh is absent (old .pt files).

  2. next_token_acc  (next-token prediction accuracy)
     Fraction of token positions t where argmax(lens(h_l_t)) == token_{t+1}.
     Rises sharply in the workspace zone and stays high into Output, then
     collapses at the last layer (which sees too little future context).
     Paper interpretation: workspace = layer that first achieves high prediction
     accuracy on the next-token task.

  3. pos_autocorr  (position autocorrelation of J-lens readout)
     Mean cosine similarity of adjacent readout distributions:
         autocorr_l = E_t[ cos(p_t, p_{t+1}) ]
     where p_t = softmax(lens(h_l_t)).
     High autocorrelation = the model is maintaining a stable concept across
     positions.  Low autocorrelation = early noise or output token-jumping.
     Peaks in the workspace zone.

  4. mean_kurtosis  (spikiness of readout distribution)
     Excess kurtosis of readout probability distribution, averaged over positions.
     High kurtosis = peaked distribution = strong single concept at each position.

  Phase assignment (4-signal majority vote):
     Each layer votes on 4 binary criteria:
       - n_active >= median_active across layers          → 1 vote
       - next_token_acc >= median_acc                     → 1 vote
       - pos_autocorr >= median_autocorr                  → 1 vote
       - mean_entropy <= min_entropy + 0.40 * entropy_range → 1 vote
     Layers with >=3 votes become Workspace candidates.
     The largest contiguous run of Workspace candidates is kept;
     isolated outliers are pruned.

  Readout formula (paper eq.):
    lens(h_l) = softmax(W_U · norm(J_l @ h_l))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy, kurtosis as scipy_kurtosis

from jspace.jlens import JacobianLens
from jspace.model import HookedModel

logger = logging.getLogger(__name__)

# gradient pursuit stops when this fraction of norm-sq is explained
_PURSUIT_COVERAGE = 0.95
# hard cap — paper reports workspace capacity of ~10–25
_MAX_K = 30


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class LayerWorkspaceStats:
    """Per-layer workspace statistics."""

    layer_idx: int
    n_active: int                           # min-k gradient pursuit (or kurtosis fallback)
    mean_entropy: float                     # mean entropy of J-lens readout distribution
    mean_kurtosis: float                    # mean excess kurtosis across positions
    next_token_acc: float                   # fraction of positions where argmax = next token
    pos_autocorr: float                     # mean cosine similarity of adjacent readout dists
    var_explained: float                    # fraction of hidden-state variance in J-space direction
    top_concepts: List[Tuple[str, float]]   # top-k concepts at last position
    phase: str = ""                         # "Early" | "Workspace" | "Output"


@dataclass
class WorkspaceReport:
    """Full workspace analysis for a single prompt."""

    prompt: str
    tokens: List[str]
    layer_stats: List[LayerWorkspaceStats]
    workspace_start: int = -1
    workspace_end: int = -1

    @property
    def workspace_layers(self) -> List[LayerWorkspaceStats]:
        return [s for s in self.layer_stats if s.phase == "Workspace"]

    @property
    def peak_capacity(self) -> int:
        ws = self.workspace_layers
        return max(s.n_active for s in ws) if ws else 0


# ── main class ────────────────────────────────────────────────────────────────

class WorkspaceAnalyzer:
    """
    Analyses global workspace properties using four paper signals:
    n_active, next_token_acc, pos_autocorr, mean_kurtosis / entropy.

    Parameters
    ----------
    jlens : JacobianLens
        Fitted J-lens (call jlens.fit() first).
    kurtosis_threshold : float
        Used only as fallback n_active detector when corpus_jh is absent.
    entropy_threshold : float | None
        Optional entropy upper bound for entropy-valley signal.
        None = auto 60% of max vocab entropy.
    var_threshold : float
        Minimum J-space variance fraction (informational; not used in phase vote).
    top_k : int
        Top-k vocabulary tokens to report per position.
    pursuit_coverage : float
        Norm-squared coverage for gradient pursuit stop criterion.
    """

    def __init__(
        self,
        jlens: JacobianLens,
        kurtosis_threshold: float = 0.0,
        entropy_threshold: Optional[float] = None,
        var_threshold: float = 0.05,
        top_k: int = 10,
        pursuit_coverage: float = _PURSUIT_COVERAGE,
    ) -> None:
        self.jlens = jlens
        self.model: HookedModel = jlens.model
        self.kurtosis_threshold = kurtosis_threshold
        self.var_threshold = var_threshold
        self.top_k = top_k
        self.pursuit_coverage = pursuit_coverage

        has_corpus = bool(jlens.corpus_jh)
        logger.info(
            "WorkspaceAnalyzer: corpus_jh=%s  n_active=%s  coverage=%.2f",
            has_corpus,
            "gradient_pursuit" if has_corpus else "kurtosis",
            pursuit_coverage,
        )

        vocab_size = self.model.vocab_size()
        max_entropy = math.log(vocab_size)
        self.entropy_threshold = (
            entropy_threshold if entropy_threshold is not None else max_entropy * 0.60
        )

    # ── public API ────────────────────────────────────────────────────────────

    def analyse(self, prompt: str, max_length: int = 128) -> WorkspaceReport:
        """Run full workspace analysis on a single prompt."""
        enc = self.model.tokenize([prompt], max_length=max_length)
        tokens = [
            self.model.tokenizer.decode([int(t)])
            for t in enc["input_ids"][0]
            if int(t) != self.model.tokenizer.pad_token_id
        ]
        token_ids = [
            int(t)
            for t in enc["input_ids"][0]
            if int(t) != self.model.tokenizer.pad_token_id
        ]

        with torch.no_grad():
            with self.model.capture_residuals(self.jlens.layer_indices) as store:
                _ = self.model.model(**enc)

        layer_stats: List[LayerWorkspaceStats] = []
        for layer_idx in self.jlens.layer_indices:
            if layer_idx not in store:
                continue
            hs = store[layer_idx]  # (1, seq, d_model)
            stats = self._compute_layer_stats(layer_idx, hs, tokens, token_ids)
            layer_stats.append(stats)

        self._assign_phases(layer_stats)
        ws_start, ws_end = self._detect_workspace_window(layer_stats)

        return WorkspaceReport(
            prompt=prompt,
            tokens=tokens,
            layer_stats=layer_stats,
            workspace_start=ws_start,
            workspace_end=ws_end,
        )

    def analyse_batch(self, prompts: List[str], max_length: int = 128) -> List[WorkspaceReport]:
        return [self.analyse(p, max_length=max_length) for p in prompts]

    # ── per-layer stats (vectorized) ──────────────────────────────────────────

    def _compute_layer_stats(
        self,
        layer_idx: int,
        hs: torch.Tensor,
        tokens: List[str],
        token_ids: List[int],
    ) -> LayerWorkspaceStats:
        """
        Compute all four workspace signals for one layer.

        hs : (1, seq, d_model)

        All seq-dimension loops are eliminated:
          - readout_logits_batch gives (seq, vocab) in one matmul + unembed
          - readout_jh_batch gives (seq, d_model) in one matmul
          - gradient_pursuit_batch vectorizes the corpus dot products
        """
        seq = hs.shape[1]
        hs_seq = hs[0]  # (seq, d_model)

        # ── readout: (seq, vocab) ──────────────────────────────────────────
        if layer_idx in self.jlens.avg_jacobians:
            logits_seq = self.jlens.readout_logits_batch(hs_seq, layer_idx)  # (seq, vocab) cpu
        else:
            logits_seq = self.model.unembed(hs_seq.unsqueeze(0)).squeeze(0).detach().cpu()

        probs_seq = F.softmax(logits_seq.float(), dim=-1)  # (seq, vocab) cpu

        # ── entropy and kurtosis ───────────────────────────────────────────
        # scipy_entropy accepts (vocab,) arrays
        probs_np = probs_seq.numpy()  # (seq, vocab)
        entropies = np.array([float(scipy_entropy(probs_np[t])) for t in range(seq)])
        kurtoses  = np.array([
            float(scipy_kurtosis(probs_np[t], fisher=True, bias=False)) for t in range(seq)
        ])
        mean_entropy  = float(entropies.mean())
        mean_kurtosis = float(kurtoses.mean())

        # ── top concepts at last position ──────────────────────────────────
        top_concepts_last = self.model.top_tokens(logits_seq[-1].to(self.model.device), k=self.top_k)

        # ── next-token accuracy ────────────────────────────────────────────
        # Compare argmax(lens(h_l_t)) with token_{t+1} for t in 0..seq-2
        next_token_acc = 0.0
        if seq >= 2 and len(token_ids) >= 2:
            preds    = logits_seq[:-1].argmax(dim=-1).numpy()   # (seq-1,)
            targets  = np.array(token_ids[1:])                  # (seq-1,) next-token ids
            n_pairs  = min(len(preds), len(targets))
            next_token_acc = float((preds[:n_pairs] == targets[:n_pairs]).mean())

        # ── position autocorrelation ───────────────────────────────────────
        # Mean cosine similarity of adjacent readout probability distributions
        pos_autocorr = 0.0
        if seq >= 2:
            # normalise each row to unit vector then dot adjacent rows
            p_norm = F.normalize(probs_seq, dim=-1)              # (seq, vocab)
            # cosine(p_t, p_{t+1}) = dot(p_norm[t], p_norm[t+1])
            dots = (p_norm[:-1] * p_norm[1:]).sum(dim=-1)        # (seq-1,)
            pos_autocorr = float(dots.mean().item())

        # ── n_active via gradient pursuit (or kurtosis fallback) ──────────
        if layer_idx in self.jlens.corpus_jh and layer_idx in self.jlens.avg_jacobians:
            jh_seq = self.jlens.readout_jh_batch(hs_seq, layer_idx)   # (seq, d_model) cpu
            k_vals = self._gradient_pursuit_batch(jh_seq, layer_idx)  # (seq,) int array
            n_active = int(round(float(k_vals.mean())))
        else:
            n_active = int((kurtoses > self.kurtosis_threshold).sum())

        # ── variance explained by J-space primary direction ────────────────
        var_explained = 0.0
        j_mat = self.jlens.avg_jacobians.get(layer_idx)
        if j_mat is not None:
            hs_flat = hs_seq.float().cpu()                       # (seq, d)
            Jh = (j_mat.float() @ hs_flat.T).T                  # (seq, d)
            j_unit = F.normalize(Jh.mean(0, keepdim=True), dim=-1).squeeze(0)  # (d,)
            proj = hs_flat @ j_unit                              # (seq,)
            var_explained = float(proj.var() / (hs_flat.var() + 1e-8))

        return LayerWorkspaceStats(
            layer_idx=layer_idx,
            n_active=n_active,
            mean_entropy=mean_entropy,
            mean_kurtosis=mean_kurtosis,
            next_token_acc=next_token_acc,
            pos_autocorr=pos_autocorr,
            var_explained=var_explained,
            top_concepts=top_concepts_last,
        )

    # ── gradient pursuit (vectorized over sequence) ───────────────────────────

    def _gradient_pursuit_batch(
        self,
        jh_seq: torch.Tensor,
        layer_idx: int,
    ) -> np.ndarray:
        """
        Sparse nonneg gradient pursuit — paper-exact active-concept counting.

        For each sequence position independently, find the minimum k unit-normed
        corpus J-lens vectors {c_i} and nonneg coefficients {a_i >= 0} such that
            ||query - sum_i a_i c_i||^2 / ||query||^2  <=  1 - pursuit_coverage

        Algorithm (nonneg matching pursuit):
          1. Pick corpus vector c* with highest POSITIVE dot product with residual.
             If max positive dot <= 0, no further nonneg progress is possible → stop.
          2. Compute coefficient  a = max(residual . c*, 0) / ||c*||^2  (nonneg clip).
          3. Subtract  a * c*  from residual.
          4. Repeat until coverage >= pursuit_coverage or _MAX_K steps reached.

        Parameters
        ----------
        jh_seq : (seq, d_model) J_l @ h for each position (CPU, float32)
        layer_idx : int

        Returns
        -------
        k_vals : (seq,) int array — number of corpus vectors needed per position
        """
        corpus = self.jlens.corpus_jh[layer_idx].float()  # (n_corpus, d_model) cpu
        seq = jh_seq.shape[0]
        k_vals = np.zeros(seq, dtype=np.int32)

        q_norms_sq = (jh_seq * jh_seq).sum(dim=-1)        # (seq,)
        residuals  = jh_seq.clone()                        # (seq, d_model)
        active     = (q_norms_sq > 1e-12).numpy()          # skip zero-norm positions

        for step in range(1, _MAX_K + 1):
            if not active.any():
                break

            active_idx = np.where(active)[0]
            r_active   = residuals[active_idx]             # (n_active, d)
            dots       = r_active @ corpus.T               # (n_active, n_corpus)

            # nonneg constraint: only positively-correlated corpus vectors are eligible
            dots_pos   = dots.clamp(min=0.0)               # (n_active, n_corpus)
            max_dots   = dots_pos.max(dim=1)               # values (n_active,)

            for i, pos in enumerate(active_idx):
                if float(max_dots.values[i]) <= 1e-12:
                    # no corpus vector has positive correlation — pursuit stalled
                    k_vals[pos] = step - 1 if step > 1 else 0
                    active[pos] = False
                    continue

                best_c_idx = int(max_dots.indices[i])
                c = corpus[best_c_idx]                     # (d,)
                c_norm_sq  = float((c * c).sum())
                if c_norm_sq < 1e-12:
                    active[pos] = False
                    continue

                # nonneg coefficient: clip to >= 0
                proj = max(float((c * residuals[pos]).sum()) / c_norm_sq, 0.0)
                residuals[pos] = residuals[pos] - proj * c

            # recompute coverage for positions that are still active
            still_active = np.where(active)[0]
            if len(still_active) == 0:
                break
            res_sq  = (residuals[still_active] * residuals[still_active]).sum(dim=-1)
            q_sq    = q_norms_sq[still_active]
            covered = 1.0 - res_sq.numpy() / (q_sq.numpy() + 1e-12)

            for i, pos in enumerate(still_active):
                if covered[i] >= self.pursuit_coverage:
                    k_vals[pos] = step
                    active[pos] = False

        # positions that never converged within _MAX_K steps
        k_vals[active] = _MAX_K
        return k_vals

    # ── phase assignment (4-signal majority vote) ─────────────────────────────

    def _assign_phases(self, stats: List[LayerWorkspaceStats]) -> None:
        """
        Label each layer Early / Workspace / Output via 4-signal majority vote.

        Output zone: final 10% of layers (next-token collapse zone) → always Output.

        For the remaining layers, each of the four signals casts a binary vote:
          1. n_active >= median(n_active)
          2. next_token_acc >= median(next_token_acc)
          3. pos_autocorr >= median(pos_autocorr)
          4. mean_entropy <= min_entropy + 0.40 * entropy_range   (valley signal)

        Workspace candidate: >= 3 votes.
        Final workspace: largest contiguous run of candidates (ties broken by
        earliest start).  Isolated single-layer candidates outside this run are
        relabelled Early.
        """
        if not stats:
            return

        n_layers = len(stats)
        # Output zone = final 10% of layers by list position (not by layer_idx, which
        # may be sparse or offset when only a subset of layers is analysed).
        # At least 1 layer is always Output; at least 1 is always non-Output.
        n_output = max(1, int(round(n_layers * 0.10)))
        output_start_pos = n_layers - n_output        # list index of first Output layer
        output_layer_ids = {s.layer_idx for s in stats[output_start_pos:]}

        non_out = [s for s in stats if s.layer_idx not in output_layer_ids]
        if not non_out:
            for s in stats:
                s.phase = "Output"
            return

        entropies   = np.array([s.mean_entropy    for s in non_out])
        n_actives   = np.array([s.n_active         for s in non_out])
        accs        = np.array([s.next_token_acc   for s in non_out])
        autocorrs   = np.array([s.pos_autocorr     for s in non_out])

        min_ent   = entropies.min()
        ent_range = max(entropies.max() - min_ent, 1e-6)
        med_active   = float(np.median(n_actives))
        med_acc      = float(np.median(accs))
        med_autocorr = float(np.median(autocorrs))

        # build vote array for non-output layers
        votes: Dict[int, int] = {}
        for i, s in enumerate(non_out):
            v  = int(n_actives[i]  >= med_active)
            v += int(accs[i]       >= med_acc)
            v += int(autocorrs[i]  >= med_autocorr)
            v += int(entropies[i]  <= min_ent + 0.40 * ent_range)
            votes[s.layer_idx] = v

        # candidates = layers with >= 3 votes
        candidate_set = {idx for idx, v in votes.items() if v >= 3}

        # find largest contiguous run of candidates among non-output layers
        ordered = [s.layer_idx for s in stats if s.layer_idx not in output_layer_ids]
        best_run: List[int] = []
        current_run: List[int] = []
        for idx in ordered:
            if idx in candidate_set:
                current_run.append(idx)
            else:
                if len(current_run) > len(best_run):
                    best_run = current_run
                current_run = []
        if len(current_run) > len(best_run):
            best_run = current_run
        workspace_set = set(best_run)

        for s in stats:
            if s.layer_idx in output_layer_ids:
                s.phase = "Output"
            elif s.layer_idx in workspace_set:
                s.phase = "Workspace"
            else:
                s.phase = "Early"

    def _detect_workspace_window(
        self, stats: List[LayerWorkspaceStats]
    ) -> Tuple[int, int]:
        ws = [s.layer_idx for s in stats if s.phase == "Workspace"]
        if not ws:
            return -1, -1
        return min(ws), max(ws)

    # ── pretty printing ───────────────────────────────────────────────────────

    @staticmethod
    def print_report(report: WorkspaceReport, max_layers: int = 30) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f'Workspace Analysis -- "{report.prompt[:60]}"')
            table.add_column("Layer",    justify="right")
            table.add_column("Phase",    justify="center")
            table.add_column("n_active", justify="right")
            table.add_column("H",        justify="right")
            table.add_column("kurt",     justify="right")
            table.add_column("nxt_acc",  justify="right")
            table.add_column("autocorr", justify="right")
            table.add_column("var_expl", justify="right")
            table.add_column("Top concept", justify="left")

            step = max(1, len(report.layer_stats) // max_layers)
            for s in report.layer_stats[::step]:
                color = {"Early": "dim", "Workspace": "green", "Output": "yellow"}.get(
                    s.phase, "white"
                )
                top_tok = s.top_concepts[0][0] if s.top_concepts else "-"
                table.add_row(
                    str(s.layer_idx),
                    f"[{color}]{s.phase}[/{color}]",
                    str(s.n_active),
                    f"{s.mean_entropy:.2f}",
                    f"{s.mean_kurtosis:.1f}",
                    f"{s.next_token_acc:.2f}",
                    f"{s.pos_autocorr:.3f}",
                    f"{s.var_explained:.3f}",
                    top_tok,
                )
            console.print(table)
            console.print(
                f"Workspace window: layers {report.workspace_start}–{report.workspace_end}  "
                f"| Peak capacity: {report.peak_capacity} active positions"
            )
        except ImportError:
            print(f"\nWorkspace Analysis: {report.prompt[:60]}")
            for s in report.layer_stats[::max(1, len(report.layer_stats) // max_layers)]:
                top = s.top_concepts[0][0] if s.top_concepts else "-"
                print(
                    f"  L{s.layer_idx:3d} [{s.phase:9s}]  "
                    f"n_active={s.n_active:2d}  H={s.mean_entropy:.2f}  "
                    f"kurt={s.mean_kurtosis:.1f}  "
                    f"nxt={s.next_token_acc:.2f}  "
                    f"ac={s.pos_autocorr:.3f}  "
                    f"var={s.var_explained:.3f}  top={top}"
                )
            print(
                f"\nWorkspace: layers {report.workspace_start}–{report.workspace_end}  "
                f"peak_capacity={report.peak_capacity}"
            )
