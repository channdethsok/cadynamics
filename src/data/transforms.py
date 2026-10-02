"""Numerical and spatial transformation functions for CADynamics.

Provides:
  - symlog: Forward transform (Physical CAD Units -> Neural Space)
  - symexp: Inverse transform (Neural Space -> Physical CAD Units)

Formula:
  symlog(x) = sign(x) * log(1 + |x|)
  symexp(y) = sign(y) * (exp(|y|) - 1)
"""

from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Forward transform (Physical CAD Units -> Neural Space) using Symmetric Logarithm.

    Maps extreme continuous CAD parameter variations (e.g. 0.5mm chamfer vs 1000mm extrusion)
    into a smooth, symmetric, zero-centered representation without precision loss or gradient saturation.

    Formula:
        symlog(x) = sign(x) * log1p(|x|)
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse transform (Neural Space -> Physical CAD Units) using Symmetric Exponential.

    Un-normalizes predicted neural continuous parameters back to physical CAD units (e.g. millimeters).

    Formula:
        symexp(x) = sign(x) * (expm1(|x|))
    """
    return torch.sign(x) * torch.expm1(torch.abs(x))