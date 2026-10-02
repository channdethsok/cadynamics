"""Backwards compatibility forwarder for src.cad.renderer."""

from __future__ import annotations

from src.cad.renderer import CANONICAL_VIEW_NAMES, HeadlessCadRenderer

__all__ = ["HeadlessCadRenderer", "CANONICAL_VIEW_NAMES"]
