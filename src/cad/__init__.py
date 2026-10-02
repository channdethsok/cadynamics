"""Native CAD & OpenCASCADE geometry processing engine for CADynamics.

Contains:
  - tracer: CadQueryRuntimeTracer for AST execution interception & solid tracking
  - brep_extractor: Boundary Representation (B-Rep) cell complex & UV-Net/BrepNet graph extraction
  - action_extractor: Action parameterization, SymLog scaling, entity selection masks & sketch wires
  - renderer: Headless offscreen 4-view canonical renderer (PyVista/VTK)
"""

from __future__ import annotations

from src.cad.action_extractor import capture_action_context, symlog
from src.cad.brep_extractor import extract_brep_state
from src.cad.renderer import HeadlessCadRenderer
from src.cad.tracer import CadQueryRuntimeTracer

__all__ = [
    "CadQueryRuntimeTracer",
    "HeadlessCadRenderer",
    "extract_brep_state",
    "capture_action_context",
    "symlog",
]
