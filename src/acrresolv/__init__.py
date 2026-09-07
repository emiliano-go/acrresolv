"""acrresolv - Hybrid acronym resolver with ML + Rules + Lookup."""

from .resolver import resolve, resolve_with_confidence, ResolveResult

__all__ = ["resolve", "resolve_with_confidence", "ResolveResult"]
