"""
Per-layer activation routing for expert Jacobians (J++ "Jacobian Filtering").

Following J++ (Ayonrinde & Lindsey, 2026): residual activations at a layer are
unit-normalised (clustering on direction, since residual norms grow with
position), centred, projected onto their top principal components, and
clustered with k-means++ / Lloyd.  Each cluster gets its own expert Jacobian
during fitting; at inference the experts are combined into one linear map, so
the router is only needed while fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


def unit_normalize(points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale each row to unit L2 norm."""
    return points / points.norm(dim=-1, keepdim=True).clamp_min(eps)


def fit_pca(points: torch.Tensor, projection_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top principal directions via the covariance eigendecomposition.

    Parameters
    ----------
    points : torch.Tensor
        (P, d) samples.
    projection_dim : int
        Number of components, in [1, d].

    Returns
    -------
    mean : torch.Tensor
        (d,) sample mean.
    components : torch.Tensor
        (d, projection_dim) principal directions in descending variance order.
    """
    n_points, d = points.shape
    if not 0 < projection_dim <= d:
        raise ValueError(f"projection_dim must be in [1, {d}], got {projection_dim}")
    mean = points.mean(0)
    centred = points - mean
    covariance = centred.T @ centred / max(n_points - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    return mean, eigenvectors[:, -projection_dim:].flip(dims=[1])


def kmeans(
    points: torch.Tensor,
    num_clusters: int,
    seed: int = 0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd's k-means with k-means++ seeding; empty clusters are re-seeded to the
    points farthest from their centroids.

    Parameters
    ----------
    points : torch.Tensor
        (P, D) samples.
    num_clusters : int
    seed : int
    max_iterations : int
    tolerance : float
        Stop once the largest centroid shift falls below this.

    Returns
    -------
    centroids : torch.Tensor
        (E, D).
    assignments : torch.Tensor
        (P,) int64 cluster indices.
    """
    n_points = points.shape[0]
    if n_points < num_clusters:
        raise ValueError(f"need >= {num_clusters} points, got {n_points}")
    generator = torch.Generator(device=points.device).manual_seed(seed)

    first = torch.randint(n_points, (1,), generator=generator, device=points.device)
    centroids = points[first]
    for _ in range(num_clusters - 1):
        nearest_sq = torch.cdist(points, centroids).pow(2).min(dim=1).values
        pick = torch.multinomial(nearest_sq.clamp_min(1e-12), 1, generator=generator)
        centroids = torch.cat([centroids, points[pick]])

    assignments = torch.zeros(n_points, dtype=torch.long, device=points.device)
    for _ in range(max_iterations):
        distances = torch.cdist(points, centroids)
        assignments = distances.argmin(dim=1)
        counts = torch.bincount(assignments, minlength=num_clusters)
        sums = torch.zeros_like(centroids).index_add_(0, assignments, points)
        updated = sums / counts.clamp_min(1).unsqueeze(1).to(points.dtype)

        empty = counts == 0
        if bool(empty.any()):
            own_distance = distances.gather(1, assignments.unsqueeze(1)).squeeze(1)
            farthest = own_distance.topk(int(empty.sum())).indices
            updated[empty] = points[farthest]

        shift = (updated - centroids).norm(dim=1).max()
        centroids = updated
        if float(shift) < tolerance:
            break
    return centroids, torch.cdist(points, centroids).argmin(dim=1)


@dataclass(frozen=True)
class ActivationRouter:
    """
    Frozen normalise → centre → project → nearest-centroid router for one layer.

    Attributes
    ----------
    mean : torch.Tensor
        (d,) PCA mean of the unit-normalised activations.
    components : torch.Tensor
        (d, projection_dim) PCA directions.
    centroids : torch.Tensor
        (E, projection_dim) k-means centroids in the projected space.
    """

    mean: torch.Tensor
    components: torch.Tensor
    centroids: torch.Tensor

    @property
    def num_clusters(self) -> int:
        """Number of experts E."""
        return int(self.centroids.shape[0])

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., d) activations → (..., projection_dim) routing coordinates."""
        dev = self.mean.device
        return (unit_normalize(activations.to(dev).float()) - self.mean) @ self.components

    def assign(self, activations: torch.Tensor) -> torch.Tensor:
        """(..., d) activations → (...) int64 expert indices."""
        projected = self.project(activations)
        flat = projected.reshape(-1, projected.shape[-1])
        return torch.cdist(flat, self.centroids).argmin(dim=1).reshape(projected.shape[:-1])

    def to_state(self) -> Dict[str, torch.Tensor]:
        """Plain-tensor state for ``torch.save(..., weights_only=True)`` round trips."""
        return {
            "mean": self.mean.cpu(),
            "components": self.components.cpu(),
            "centroids": self.centroids.cpu(),
        }

    @classmethod
    def from_state(
        cls, state: Dict[str, torch.Tensor], device: torch.device | str
    ) -> "ActivationRouter":
        """Inverse of :meth:`to_state`."""
        return cls(**{k: v.to(device=device, dtype=torch.float32) for k, v in state.items()})


def fit_router(
    activations: torch.Tensor,
    num_clusters: int,
    projection_dim: int,
    seed: int = 0,
) -> ActivationRouter:
    """
    Fit one layer's router.

    Parameters
    ----------
    activations : torch.Tensor
        (P, d) residual activations at the layer.
    num_clusters : int
    projection_dim : int
    seed : int

    Returns
    -------
    ActivationRouter
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be (P, d), got {tuple(activations.shape)}")
    normed = unit_normalize(activations.float())
    mean, components = fit_pca(normed, projection_dim)
    centroids, _ = kmeans((normed - mean) @ components, num_clusters, seed=seed)
    return ActivationRouter(mean=mean, components=components, centroids=centroids)
