"""jspace — Jacobian Lens and Global Workspace replication for open LLMs."""

from jspace.model import HookedModel
from jspace.jlens import JacobianLens
from jspace.workspace import WorkspaceAnalyzer
from jspace.offset_lens import OffsetJacobianLens, OffsetLensConfig
from jspace.multitoken import MultiTokenReport, MultiTokenWorkspace

__all__ = [
    "HookedModel",
    "JacobianLens",
    "WorkspaceAnalyzer",
    "OffsetJacobianLens",
    "OffsetLensConfig",
    "MultiTokenReport",
    "MultiTokenWorkspace",
]
