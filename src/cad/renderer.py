"""Deterministic headless in-memory multi-view renderer for Action-JEPA-CAD.

Renders B-Rep geometry directly in memory into [4, 3, 224, 224] uint8 tensors
representing 4 canonical perspectives:
  0: Isometric (3D spatial overview)
  1: Front (XZ plane projection)
  2: Top (XY plane projection)
  3: Right (YZ plane projection)

Features:
  - Robust Mesa/OpenGL off-screen context initialization before OCP/VTK conflicts
  - Clean CAD feature edge extraction (eliminates triangulation artifacts on flat faces)
  - Studio lighting and tight zoom framing
  - Zero disk I/O (pure in-memory tensor generation)
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pyvista as pv
import torch
import vtk

# Suppress VTK hardware capability warnings in headless container environments
vtk.vtkObject.GlobalWarningDisplayOff()
logger = logging.getLogger(__name__)

CANONICAL_VIEW_NAMES: Tuple[str, ...] = ("iso", "front", "top", "right")

# Pre-initialize PyVista off-screen context to establish OpenGL handshake with Mesa
try:
    _init_plotter = pv.Plotter(off_screen=True, window_size=[10, 10])
    _init_plotter.render()
    _init_plotter.screenshot(transparent_background=False)
    _init_plotter.close()
except Exception:
    pass


class HeadlessCadRenderer:
    """Headless off-screen renderer producing deterministic 4-view uint8 tensors."""

    def __init__(self, image_size: int = 224, linear_deflection: float = 0.05) -> None:
        self.image_size = image_size
        self.linear_deflection = linear_deflection
        self._ensure_gl_init()

    def _ensure_gl_init(self) -> None:
        try:
            p = pv.Plotter(off_screen=True, window_size=[10, 10])
            p.render()
            p.screenshot(transparent_background=False)
            p.close()
        except Exception:
            pass

    def render_empty(self) -> torch.Tensor:
        """Return deterministic neutral gray canvas for empty workspace S_0 across 4 views."""
        return torch.full((4, 3, self.image_size, self.image_size), 220, dtype=torch.uint8)

    def render(
        self,
        shape_obj: Any,
        reference_bbox: Optional[Union[Tuple[float, ...], List[float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Render CadQuery Shape / Solid / Compound in-memory to [4, 3, 224, 224] uint8.

        Captures 4 canonical perspectives (Isometric, Front, Top, Right).
        When reference_bbox is provided, locks camera framing to fixed metric volume.
        """
        if shape_obj is None:
            return self.render_empty()

        try:
            # Extract tessellatable solid
            if hasattr(shape_obj, "toCompound"):
                solid = shape_obj.toCompound()
            elif hasattr(shape_obj, "val"):
                solid = shape_obj.val()
            else:
                solid = shape_obj

            if solid is None or not hasattr(solid, "tessellate"):
                return self.render_empty()

            verts, tris = solid.tessellate(self.linear_deflection)
            if len(verts) == 0 or len(tris) == 0:
                return self.render_empty()

            v_arr = np.array([[v.x, v.y, v.z] for v in verts], dtype=np.float32)
            tris_arr = np.array(tris, dtype=np.int32)
            mesh = pv.PolyData.from_regular_faces(v_arr, tris_arr)

            # Extract clean CAD feature edges (hides planar face triangulation diagonals)
            feature_edges = mesh.extract_feature_edges(
                boundary_edges=True,
                non_manifold_edges=True,
                feature_angle=30.0,
                manifold_edges=False,
            )

            # Convert reference_bbox [xmin, ymin, zmin, xmax, ymax, zmax] to PyVista bounds
            bounds = None
            if reference_bbox is not None:
                if isinstance(reference_bbox, torch.Tensor):
                    b = reference_bbox.detach().cpu().tolist()
                else:
                    b = list(reference_bbox)
                if len(b) == 6:
                    xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in b]
                    # PyVista expects (xmin, xmax, ymin, ymax, zmin, zmax)
                    dx = max(1e-3, (xmax - xmin) * 0.05)
                    dy = max(1e-3, (ymax - ymin) * 0.05)
                    dz = max(1e-3, (zmax - zmin) * 0.05)
                    bounds = (xmin - dx, xmax + dx, ymin - dy, ymax + dy, zmin - dz, zmax + dz)

            plotter = pv.Plotter(off_screen=True, window_size=[self.image_size, self.image_size])
            plotter.set_background("white")

            # Add surface material
            plotter.add_mesh(
                mesh,
                color="#CBD5E1",
                smooth_shading=True,
                show_edges=False,
            )
            # Add prominent CAD feature edge outlines
            plotter.add_mesh(
                feature_edges,
                color="#1A1A1A",
                line_width=2.0,
            )

            views_imgs: List[torch.Tensor] = []

            # 1. Isometric View
            plotter.camera_position = "iso"
            if bounds is not None:
                plotter.reset_camera(bounds=bounds)
            else:
                plotter.reset_camera()
            plotter.camera.zoom(0.85)
            plotter.render()
            img_iso = plotter.screenshot(return_img=True, transparent_background=False)
            views_imgs.append(self._to_chw_tensor(img_iso))

            # 2. Front View (XZ plane, looking along -Y)
            plotter.view_xz()
            if bounds is not None:
                plotter.reset_camera(bounds=bounds)
            else:
                plotter.reset_camera()
            plotter.camera.zoom(0.85)
            plotter.render()
            img_front = plotter.screenshot(return_img=True, transparent_background=False)
            views_imgs.append(self._to_chw_tensor(img_front))

            # 3. Top View (XY plane, looking along -Z)
            plotter.view_xy()
            if bounds is not None:
                plotter.reset_camera(bounds=bounds)
            else:
                plotter.reset_camera()
            plotter.camera.zoom(0.85)
            plotter.render()
            img_top = plotter.screenshot(return_img=True, transparent_background=False)
            views_imgs.append(self._to_chw_tensor(img_top))

            # 4. Right View (YZ plane, looking along -X)
            plotter.view_yz()
            if bounds is not None:
                plotter.reset_camera(bounds=bounds)
            else:
                plotter.reset_camera()
            plotter.camera.zoom(0.85)
            plotter.render()
            img_right = plotter.screenshot(return_img=True, transparent_background=False)
            views_imgs.append(self._to_chw_tensor(img_right))

            plotter.close()

            # Stack into [4, 3, H, W] uint8
            return torch.stack(views_imgs, dim=0)

        except Exception as e:
            logger.debug(f"Headless rendering error: {e}")
            return self.render_empty()

    def _to_chw_tensor(self, img_np: Optional[np.ndarray]) -> torch.Tensor:
        """Convert numpy HWC image array to CHW uint8 PyTorch tensor."""
        if img_np is None or img_np.ndim != 3:
            return torch.full((3, self.image_size, self.image_size), 220, dtype=torch.uint8)
        return torch.from_numpy(img_np[:, :, :3]).permute(2, 0, 1).contiguous()
