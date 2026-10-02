"""Backwards compatibility forwarder for src.cad.brep_extractor."""

from __future__ import annotations

from src.cad.brep_extractor import (
    _extract_edges_data,
    extract_brep_state,
)

__all__ = ["extract_brep_state", "_extract_edges_data"]
