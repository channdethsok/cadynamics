"""Backwards compatibility forwarder for src.cad.action_extractor."""

from __future__ import annotations

from src.cad.action_extractor import (
    capture_action_context,
    symexp,
    symlog,
)

__all__ = ["capture_action_context", "symlog", "symexp"]
