"""
Multi-token workspace analysis on top of an offset-resolved J-lens.

For every analysed layer and token position the offset lens yields one
vocabulary distribution per future offset Δ — a latent string.  Per layer this
module reports:

  * latent strings      top-k tokens for every (Δ, position)
  * lookahead accuracy  fraction of positions t whose offset-Δ argmax equals the
                        actual token t+Δ+1 (Δ=0 is the single-token lens's
                        next-token accuracy signal)
  * readout entropy     mean entropy of each offset's distribution
  * concept counts      nonnegative matching pursuit against the corpus atoms,
                        jointly across offsets (a multi-token concept costs one
                        atom) and split per offset (each offset explained on its
                        own, counts summed).  Joint coverage is measured on the
                        stacked vector, i.e. averaged over offsets; split coverage
                        is required per offset.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from jspace.model import HookedModel
from jspace.offset_lens import OffsetJacobianLens
from jspace.pursuit import nonneg_matching_pursuit

logger = logging.getLogger(__name__)


def lookahead_accuracy(
    pred_ids: torch.Tensor,
    token_ids: torch.Tensor,
    offsets: Sequence[int],
) -> torch.Tensor:
    """
    Top-1 accuracy of predicting token t+Δ+1 from position t, per offset.

    Parameters
    ----------
    pred_ids : torch.Tensor
        (K, T) argmax token ids of the offset readouts.
    token_ids : torch.Tensor
        (T,) actual token ids of the sequence.
    offsets : Sequence[int]
        K offsets aligned with ``pred_ids`` rows.

    Returns
    -------
    torch.Tensor
        (K,) float32 accuracy; NaN for offsets with no in-sequence target.
    """
    n_offsets, seq_len = pred_ids.shape
    if len(offsets) != n_offsets:
        raise ValueError(f"{len(offsets)} offsets for {n_offsets} prediction rows")
    if token_ids.shape != (seq_len,):
        raise ValueError(f"token_ids shape {tuple(token_ids.shape)} != ({seq_len},)")

    device = pred_ids.device
    offset_col = torch.as_tensor(list(offsets), device=device).unsqueeze(1)
    target_pos = torch.arange(seq_len, device=device).unsqueeze(0) + offset_col + 1
    valid = target_pos < seq_len
    targets = token_ids.to(device)[target_pos.clamp(max=seq_len - 1)]
    hits = ((pred_ids == targets) & valid).sum(-1).float()
    n_valid = valid.sum(-1).float()
    return torch.where(
        n_valid > 0, hits / n_valid.clamp_min(1.0), torch.full_like(hits, float("nan"))
    )


@dataclass
class LayerLatentStats:
    """
    Multi-token workspace statistics for one layer of one prompt.

    Attributes
    ----------
    layer_idx : int
    top_ids : torch.Tensor
        (K, T, k) top-k token ids per offset and position.
    top_probs : torch.Tensor
        (K, T, k) matching probabilities.
    entropy : torch.Tensor
        (K,) mean readout entropy (nats) per offset.
    lookahead_acc : torch.Tensor
        (K,) top-1 accuracy against the actual token Δ+1 ahead; NaN if undefined.
    n_joint : float | None
        Mean distinct atoms per position, pursuit on offset-stacked vectors.
    joint_converged : float | None
        Fraction of positions whose joint pursuit reached coverage.
    n_split : float | None
        Mean over positions of the per-offset atom counts summed over offsets.
    split_converged : float | None
        Fraction of (offset, position) pursuits that reached coverage.
    """

    layer_idx: int
    top_ids: torch.Tensor
    top_probs: torch.Tensor
    entropy: torch.Tensor
    lookahead_acc: torch.Tensor
    n_joint: Optional[float] = None
    joint_converged: Optional[float] = None
    n_split: Optional[float] = None
    split_converged: Optional[float] = None


@dataclass
class MultiTokenReport:
    """
    Multi-token workspace analysis of a single prompt.

    Attributes
    ----------
    prompt : str
    token_ids : list[int]
    tokens : list[str]
    offsets : list[int]
    layer_stats : list[LayerLatentStats]
    baseline_acc : torch.Tensor
        (K,) accuracy of the model's own next-token argmax at t against token
        t+Δ+1.  At Δ=0 this is the model's output accuracy (the ceiling for the
        lens); at Δ>0 it is a repetition baseline the offset readout must beat.
    """

    prompt: str
    token_ids: List[int]
    tokens: List[str]
    offsets: List[int]
    layer_stats: List[LayerLatentStats]
    baseline_acc: torch.Tensor

    def layer(self, layer_idx: int) -> LayerLatentStats:
        """Return the statistics of one analysed layer."""
        for stats in self.layer_stats:
            if stats.layer_idx == layer_idx:
                return stats
        raise KeyError(f"Layer {layer_idx} not in report {[s.layer_idx for s in self.layer_stats]}")


class MultiTokenWorkspace:
    """
    Reads multi-token latent strings and concept counts from an offset J-lens.

    Parameters
    ----------
    lens : OffsetJacobianLens
        Fitted lens.
    top_k : int
        Tokens kept per (offset, position).
    count_concepts : bool
        Run joint and split matching pursuit (requires ``lens.corpus_atoms``).
    coverage : float
        Pursuit coverage target.
    max_atoms : int
        Pursuit step cap.
    """

    def __init__(
        self,
        lens: OffsetJacobianLens,
        top_k: int = 5,
        count_concepts: bool = True,
        coverage: float = 0.95,
        max_atoms: int = 30,
    ) -> None:
        if not lens.jacobians:
            raise ValueError("lens is not fitted")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if count_concepts and not (lens.corpus_atoms and lens.corpus_mean):
            raise ValueError(
                "count_concepts=True needs corpus atoms and mean; call build_corpus_atoms()"
            )
        self.lens = lens
        self.model: HookedModel = lens.model
        self.top_k = top_k
        self.count_concepts = count_concepts
        self.coverage = coverage
        self.max_atoms = max_atoms

    # ── analysis ────────────────────────────────────────────────────────────

    def analyse(self, prompt: str) -> MultiTokenReport:
        """
        Analyse one prompt.

        Parameters
        ----------
        prompt : str

        Returns
        -------
        MultiTokenReport
        """
        enc = self.model.tokenize([prompt], max_length=self.lens.config.max_length)
        return self.analyse_encoding(enc, prompt=prompt)

    def analyse_encoding(
        self, enc: Mapping[str, torch.Tensor], prompt: str = ""
    ) -> MultiTokenReport:
        """
        Analyse one pre-tokenised, unpadded sequence.

        Parameters
        ----------
        enc : Mapping[str, torch.Tensor]
            Model inputs with ``input_ids`` of shape (1, T).
        prompt : str
            Source text, recorded in the report.

        Returns
        -------
        MultiTokenReport
        """
        input_ids = enc["input_ids"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"expected input_ids of shape (1, T), got {tuple(input_ids.shape)}")
        mask = enc.get("attention_mask")
        if mask is not None and not bool(mask.all()):
            raise ValueError("padded sequences are not supported")
        token_ids = input_ids[0]

        with torch.no_grad():
            with self.model.capture_residuals(self.lens.layer_indices) as store:
                logits = self.model.model(**enc, use_cache=False).logits[0]
            next_token = logits.argmax(-1).unsqueeze(0).expand(len(self.lens.offsets), -1)
            baseline_acc = lookahead_accuracy(next_token, token_ids, self.lens.offsets).cpu()
            del logits
            layer_stats = [
                self._layer_stats(layer, store[layer][0], token_ids)
                for layer in self.lens.layer_indices
            ]

        ids = token_ids.tolist()
        return MultiTokenReport(
            prompt=prompt,
            token_ids=ids,
            tokens=[self.model.tokenizer.decode([i]) for i in ids],
            offsets=list(self.lens.offsets),
            layer_stats=layer_stats,
            baseline_acc=baseline_acc,
        )

    def latent_string(
        self,
        report: MultiTokenReport,
        layer_idx: int,
        position: int = -1,
        rank: int = 0,
    ) -> List[Tuple[int, str, float]]:
        """
        Decode the latent string at one layer and position.

        Parameters
        ----------
        report : MultiTokenReport
        layer_idx : int
        position : int
            Token position; negative values index from the end.
        rank : int
            Which of the top-k candidates to decode (0 = argmax).

        Returns
        -------
        list[tuple[int, str, float]]
            (offset, token, probability) for every offset.
        """
        stats = report.layer(layer_idx)
        if not 0 <= rank < stats.top_ids.shape[-1]:
            raise ValueError(f"rank must be in [0, {stats.top_ids.shape[-1]}), got {rank}")
        ids = stats.top_ids[:, position, rank].tolist()
        probs = stats.top_probs[:, position, rank].tolist()
        return [
            (delta, self.model.tokenizer.decode([tok]), prob)
            for delta, tok, prob in zip(report.offsets, ids, probs)
        ]

    def to_summary(self, report: MultiTokenReport, position: int = -1) -> Dict[str, Any]:
        """
        JSON-serialisable summary of a report.

        Parameters
        ----------
        report : MultiTokenReport
        position : int
            Position whose latent strings are decoded.

        Returns
        -------
        dict
        """

        def clean(values: torch.Tensor) -> List[Optional[float]]:
            return [None if math.isnan(v) else round(v, 5) for v in values.tolist()]

        layers = []
        for stats in report.layer_stats:
            candidates = [
                self.latent_string(report, stats.layer_idx, position, rank)
                for rank in range(stats.top_ids.shape[-1])
            ]
            layers.append(
                {
                    "layer": stats.layer_idx,
                    "lookahead_acc": clean(stats.lookahead_acc),
                    "entropy": clean(stats.entropy),
                    "n_joint": stats.n_joint,
                    "joint_converged": stats.joint_converged,
                    "n_split": stats.n_split,
                    "split_converged": stats.split_converged,
                    "latent_string": [
                        {"offset": delta, "top": [[c[k][1], round(c[k][2], 5)] for c in candidates]}
                        for k, delta in enumerate(report.offsets)
                    ],
                }
            )
        return {
            "prompt": report.prompt,
            "tokens": report.tokens,
            "offsets": report.offsets,
            "baseline_acc": clean(report.baseline_acc),
            "latent_string_position": position,
            "layers": layers,
        }

    def print_report(self, report: MultiTokenReport, max_layers: int = 40) -> None:
        """
        Render a per-layer table of lookahead accuracy, concept counts, and the
        latent string at the last position.

        Parameters
        ----------
        report : MultiTokenReport
        max_layers : int
            Subsample layers evenly above this many rows.
        """
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table

        table = Table(title=f'Multi-token J-space — "{escape(report.prompt[:60])}"')
        table.add_column("Layer", justify="right")
        table.add_column("acc Δ=" + ",".join(str(o) for o in report.offsets), justify="left")
        table.add_column("n_joint (conv)", justify="right")
        table.add_column("n_split (conv)", justify="right")
        table.add_column("Latent string @ last token (top-1 per Δ)", justify="left")

        step = max(1, math.ceil(len(report.layer_stats) / max_layers))
        for stats in report.layer_stats[::step]:
            accs = " ".join(
                "  - " if math.isnan(a) else f"{a:.2f}" for a in stats.lookahead_acc.tolist()
            )
            latent = " │ ".join(
                f"{escape(tok.strip() or repr(tok))} [dim]{prob:.2f}[/dim]"
                for _, tok, prob in self.latent_string(report, stats.layer_idx)
            )
            table.add_row(
                str(stats.layer_idx),
                accs,
                (
                    "-"
                    if stats.n_joint is None
                    else f"{stats.n_joint:.1f} ({stats.joint_converged:.0%})"
                ),
                (
                    "-"
                    if stats.n_split is None
                    else f"{stats.n_split:.1f} ({stats.split_converged:.0%})"
                ),
                latent,
            )
        Console().print(table)
        Console().print(
            "baseline (model's own next-token argmax vs token t+Δ+1): "
            + " ".join("-" if math.isnan(a) else f"{a:.2f}" for a in report.baseline_acc.tolist())
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _layer_stats(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> LayerLatentStats:
        """Compute readout, lookahead, entropy, and concept statistics for one layer."""
        projected = self.lens.project(hidden, layer_idx)
        log_probs = F.log_softmax(self.model.unembed(projected).float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean(-1)
        top_logp, top_ids = log_probs.topk(self.top_k, dim=-1)
        del log_probs

        stats = LayerLatentStats(
            layer_idx=layer_idx,
            top_ids=top_ids.cpu(),
            top_probs=top_logp.exp().cpu(),
            entropy=entropy.cpu(),
            lookahead_acc=lookahead_accuracy(top_ids[..., 0], token_ids, self.lens.offsets).cpu(),
        )
        if self.count_concepts:
            self._count_concepts(projected, layer_idx, stats)
        return stats

    def _count_concepts(
        self,
        projected: torch.Tensor,
        layer_idx: int,
        stats: LayerLatentStats,
    ) -> None:
        """Fill joint and split concept counts on ``stats`` in place."""
        n_offsets, seq_len, d = projected.shape
        atoms = self.lens.corpus_atoms[layer_idx].to(device=projected.device, dtype=torch.float32)
        mean = self.lens.corpus_mean[layer_idx].to(projected.device).unsqueeze(1)
        # Each offset block is centred on the corpus mean like the atoms, then unit-normalised
        # before stacking; otherwise the Δ=0 block, an order of magnitude larger, would
        # dominate the joint pursuit.
        blocks = F.normalize(projected - mean, dim=-1)

        joint = nonneg_matching_pursuit(
            blocks.transpose(0, 1).reshape(1, seq_len, n_offsets * d),
            atoms.reshape(1, -1, n_offsets * d),
            coverage=self.coverage,
            max_steps=self.max_atoms,
        )
        split = nonneg_matching_pursuit(
            blocks,
            atoms.transpose(0, 1),
            coverage=self.coverage,
            max_steps=self.max_atoms,
        )
        stats.n_joint = float(joint.n_atoms.float().mean())
        stats.joint_converged = float(joint.converged.float().mean())
        stats.n_split = float(split.n_atoms.sum(0).float().mean())
        stats.split_converged = float(split.converged.float().mean())
