"""Shared backend contract types for optional accelerator backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ["BackendCapabilities", "MesherBackend"]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Capability handshake returned by backend implementations."""

    supports_cuda: bool
    supports_mps: bool
    preferred_dtype: str
    max_batch: int
    supports_dlpack: bool
    supports_async_streams: bool
    version: str


@runtime_checkable
class MesherBackend(Protocol):
    """Protocol for backends that expose batched SDF evaluation."""

    BACKEND_VERSION: str

    def get_capabilities(self) -> BackendCapabilities:
        """Return backend availability and preferred execution policy."""

    def evaluate_sdf_batch(self, kernels: Any, positions: Any, **kwargs: Any) -> Any:
        """Evaluate a large SDF batch using backend-native tensors where possible."""

    def to_backend_dlpack(self, positions: Any) -> Any:
        """Export backend-native positions as a DLPack capsule."""
