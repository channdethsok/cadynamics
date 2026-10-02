"""Action extractor capturing CAD operation parameters, references, and sketch geometry.

Captured BEFORE topology-modifying operations are executed to ensure:
  - Reference entities match S_k face graph indices via occ_shape_a.IsSame(occ_shape_b)
  - Pending sketch wires are preserved before CadQuery consumes them
  - Spatial anchors and metric parameters are causally normalized
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import cadquery as cq

import OCP.BRepAdaptor as BRepAdaptor
import OCP.BRepGProp as BRepGProp
import OCP.GProp as GProp
import OCP.gp as gp
import OCP.GeomAbs as GeomAbs
import OCP.ShapeAnalysis as ShapeAnalysis
import OCP.TopAbs as TopAbs
import OCP.TopExp as TopExp
import OCP.TopoDS as TopoDS
import OCP.TopTools as TopTools

from src.data.vocabulary import (
    CMD2ID,
    REF_EDGE,
    REF_FACE,
    REF_NONE,
    REF_VERTEX,
    REF_WORKPLANE,
    get_command_id,
)
from src.data.transforms import symexp, symlog

logger = logging.getLogger(__name__)


def capture_action_context(
    workplane: Any,
    operation_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    prev_state_shape: Any,
) -> Dict[str, Any]:
    """Capture complete action information BEFORE executing operation on workplane.

    Args:
        workplane: The cadquery.Workplane instance prior to method execution.
        operation_name: Name of solid-commit method (e.g. 'extrude', 'hole', 'chamfer').
        args: Positional arguments passed to the method.
        kwargs: Keyword arguments passed to the method.
        prev_state_shape: The TopoDS_Shape or Solid corresponding to S_k.

    Returns:
        action: Fully populated and validated Action dictionary.
    """
    cmd_id = get_command_id(operation_name)
    if cmd_id is None:
        cmd_id = CMD2ID.get("OTHER", 15)

    # 1. Parse raw metric arguments from args and kwargs
    raw_metric_nums = _extract_raw_numeric_args(operation_name, args, kwargs)

    # 2. Extract reference anchors and selected entities from workplane
    ref_kind, ref_indices, ref_points, ref_dirs, selected_faces_mask, selected_edges_mask = _extract_references(
        workplane=workplane,
        prev_state_shape=prev_state_shape,
    )

    # 3. Construct 12-D action summary (causally normalized)
    params_raw, params, param_mask = _build_12d_summary(
        ref_kind=ref_kind,
        ref_points=ref_points,
        ref_dirs=ref_dirs,
        metric_nums=raw_metric_nums,
        workplane=workplane,
    )

    # 4. Extract pending sketch profile geometry before CadQuery clears it
    profile_meta = _extract_pending_profile(workplane)

    # 5. Extract path or tool geometry if applicable
    path_meta = None
    tool_meta = None
    if operation_name.lower() in ("union", "fuse", "intersect") and len(args) > 0:
        tool_meta = {"tool_type": str(type(args[0]))}

    return {
        "cmd_id": cmd_id,
        "params_raw": params_raw,
        "params": params,
        "param_mask": param_mask,
        "ref_kind": ref_kind,
        "ref_entity_indices": ref_indices,
        "ref_points": ref_points,
        "ref_directions": ref_dirs,
        "selected_faces_mask": selected_faces_mask,
        "selected_edges_mask": selected_edges_mask,
        "profile": profile_meta,
        "path": path_meta,
        "tool_geometry": tool_meta,
        "operation_name": operation_name,
    }


def _extract_references(
    workplane: Any,
    prev_state_shape: Any,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inspect active workplane and selected entities to resolve spatial references."""
    selected_objects = list(workplane.objects) if hasattr(workplane, "objects") else []

    # Map previous state faces and edges for exact IsSame topological comparison
    prev_face_map = TopTools.TopTools_IndexedMapOfShape()
    prev_edge_map = TopTools.TopTools_IndexedMapOfShape()
    edge_to_faces = TopTools.TopTools_IndexedDataMapOfShapeListOfShape()

    if prev_state_shape is not None:
        occ_shape = (
            prev_state_shape.toCompound().wrapped
            if hasattr(prev_state_shape, "toCompound")
            else (
                prev_state_shape.val().wrapped
                if hasattr(prev_state_shape, "val")
                else (
                    prev_state_shape.wrapped
                    if hasattr(prev_state_shape, "wrapped")
                    else prev_state_shape
                )
            )
        )
        if occ_shape is not None and not (
            hasattr(occ_shape, "IsNull") and occ_shape.IsNull()
        ):
            TopExp.TopExp.MapShapes_s(
                occ_shape, TopAbs.TopAbs_ShapeEnum.TopAbs_FACE, prev_face_map
            )
            TopExp.TopExp.MapShapes_s(
                occ_shape, TopAbs.TopAbs_ShapeEnum.TopAbs_EDGE, prev_edge_map
            )
            TopExp.TopExp.MapShapesAndAncestors_s(
                occ_shape,
                TopAbs.TopAbs_ShapeEnum.TopAbs_EDGE,
                TopAbs.TopAbs_ShapeEnum.TopAbs_FACE,
                edge_to_faces,
            )

    n_prev_faces = prev_face_map.Size()
    n_prev_edges = prev_edge_map.Size()
    selected_faces_mask = torch.zeros(n_prev_faces, dtype=torch.bool)
    selected_edges_mask = torch.zeros(n_prev_edges, dtype=torch.bool)

    matched_indices: List[int] = []
    ref_points_list: List[List[float]] = []
    ref_dirs_list: List[List[float]] = []
    ref_kind = REF_NONE

    target_objects = list(selected_objects)
    # If selected_objects only contains sketch geometry (Wires) or is empty,
    # inspect parent workplanes for the underlying topological reference (e.g. faces(">Z").workplane().circle(...))
    if not target_objects or all(
        hasattr(o, "wrapped") and isinstance(o.wrapped, (TopoDS.TopoDS_Wire, TopoDS.TopoDS_Compound))
        for o in target_objects
    ):
        parent_cursor = getattr(workplane, "parent", None)
        while parent_cursor is not None:
            p_objs = getattr(parent_cursor, "objects", [])
            if p_objs and any(
                hasattr(o, "wrapped")
                and isinstance(o.wrapped, (TopoDS.TopoDS_Face, TopoDS.TopoDS_Edge, TopoDS.TopoDS_Vertex))
                for o in p_objs
            ):
                target_objects = list(p_objs)
                break
            parent_cursor = getattr(parent_cursor, "parent", None)

    if len(target_objects) > 0:
        first_obj = target_objects[0]

        # Case A: Selected Faces
        if isinstance(first_obj, cq.Face) or (
            hasattr(first_obj, "wrapped")
            and isinstance(first_obj.wrapped, TopoDS.TopoDS_Face)
        ):
            ref_kind = REF_FACE
            for obj in target_objects:
                occ_face = obj.wrapped if hasattr(obj, "wrapped") else obj
                if hasattr(occ_face, "Orientation"):
                    # Find matching face index in S_k strictly using IsSame
                    f_idx = -1
                    for k in range(1, prev_face_map.Size() + 1):
                        if occ_face.IsSame(prev_face_map.FindKey(k)):
                            f_idx = k - 1
                            break

                    matched_indices.append(f_idx if f_idx >= 0 else 0)
                    if 0 <= f_idx < n_prev_faces:
                        selected_faces_mask[f_idx] = True

                    # Extract face centroid and normal
                    props = GProp.GProp_GProps()
                    BRepGProp.BRepGProp.SurfaceProperties_s(occ_face, props)
                    com = props.CentreOfMass()
                    ref_points_list.append(
                        [float(com.X()), float(com.Y()), float(com.Z())]
                    )

                    try:
                        adaptor = BRepAdaptor.BRepAdaptor_Surface(occ_face)
                        pnt = gp.gp_Pnt()
                        d1u = gp.gp_Vec()
                        d1v = gp.gp_Vec()
                        adaptor.D1(
                            0.5
                            * (adaptor.FirstUParameter() + adaptor.LastUParameter()),
                            0.5
                            * (adaptor.FirstVParameter() + adaptor.LastVParameter()),
                            pnt,
                            d1u,
                            d1v,
                        )
                        n_vec = d1u.Crossed(d1v)
                        if n_vec.Magnitude() > 1e-6:
                            n_vec.Normalize()
                            norm = [
                                float(n_vec.X()),
                                float(n_vec.Y()),
                                float(n_vec.Z()),
                            ]
                        else:
                            norm = [0.0, 0.0, 1.0]
                    except Exception:
                        norm = [0.0, 0.0, 1.0]

                    if (
                        occ_face.Orientation()
                        == TopAbs.TopAbs_Orientation.TopAbs_REVERSED
                    ):
                        norm = [-n for n in norm]
                    ref_dirs_list.append(norm)

        # Case B: Selected Edges (e.g. for fillet / chamfer)
        elif hasattr(first_obj, "wrapped") and isinstance(
            first_obj.wrapped, TopoDS.TopoDS_Edge
        ):
            ref_kind = REF_EDGE
            for obj in target_objects:
                occ_edge = obj.wrapped if hasattr(obj, "wrapped") else obj
                # Find matching physical edge in prev_edge_map
                e_idx = -1
                for k in range(1, prev_edge_map.Size() + 1):
                    if occ_edge.IsSame(prev_edge_map.FindKey(k)):
                        e_idx = k - 1
                        break
                if 0 <= e_idx < n_prev_edges:
                    selected_edges_mask[e_idx] = True

                # Find incident face index
                matched_face_idx = 0
                for e_i in range(1, edge_to_faces.Size() + 1):
                    if occ_edge.IsSame(edge_to_faces.FindKey(e_i)):
                        flist = edge_to_faces.FindFromIndex(e_i)
                        for f in flist:
                            matched_face_idx = prev_face_map.FindIndex(f) - 1
                            break
                        break
                matched_indices.append(max(0, matched_face_idx))

                # Midpoint of edge
                try:
                    c = obj.Center()
                    ref_points_list.append([float(c.x), float(c.y), float(c.z)])
                except Exception:
                    ref_points_list.append([0.0, 0.0, 0.0])

                # Tangent direction (deterministic orientation)
                try:
                    t_dir = obj.tangentAt(0.5)
                    ref_dirs_list.append(
                        [float(t_dir.x), float(t_dir.y), float(t_dir.z)]
                    )
                except Exception:
                    ref_dirs_list.append([0.0, 0.0, 1.0])

        # Case C: Selected Vertices
        elif hasattr(first_obj, "wrapped") and isinstance(
            first_obj.wrapped, TopoDS.TopoDS_Vertex
        ):
            ref_kind = REF_VERTEX
            for obj in target_objects:
                try:
                    c = obj.Center()
                    ref_points_list.append([float(c.x), float(c.y), float(c.z)])
                except Exception:
                    ref_points_list.append([0.0, 0.0, 0.0])
                ref_dirs_list.append([0.0, 0.0, 1.0])
                matched_indices.append(0)

    # Case D: Active Workplane
    if ref_kind == REF_NONE:
        plane = getattr(workplane, "plane", None)
        if plane is not None:
            ref_kind = REF_WORKPLANE
            try:
                orig = plane.origin
                zdir = plane.zDir
                ref_points_list.append([float(orig.x), float(orig.y), float(orig.z)])
                ref_dirs_list.append([float(zdir.x), float(zdir.y), float(zdir.z)])
            except Exception:
                ref_points_list.append([0.0, 0.0, 0.0])
                ref_dirs_list.append([0.0, 0.0, 1.0])
            matched_indices.append(0)

    # Empty fallback
    if len(ref_points_list) == 0:
        ref_kind = REF_NONE
        ref_indices_tensor = torch.zeros(0, dtype=torch.long)
        ref_points_tensor = torch.zeros((0, 3), dtype=torch.float32)
        ref_dirs_tensor = torch.zeros((0, 3), dtype=torch.float32)
    else:
        ref_indices_tensor = torch.tensor(matched_indices, dtype=torch.long)
        ref_points_tensor = torch.tensor(ref_points_list, dtype=torch.float32)
        ref_dirs_tensor = torch.tensor(ref_dirs_list, dtype=torch.float32)

    return (
        ref_kind,
        ref_indices_tensor,
        ref_points_tensor,
        ref_dirs_tensor,
        selected_faces_mask,
        selected_edges_mask,
    )


def _build_12d_summary(
    ref_kind: int,
    ref_points: torch.Tensor,
    ref_dirs: torch.Tensor,
    metric_nums: List[float],
    workplane: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct deterministic 12-dimensional action summary."""
    params_raw = torch.zeros(12, dtype=torch.float32)
    params = torch.zeros(12, dtype=torch.float32)
    mask = torch.zeros(12, dtype=torch.bool)

    # Dims 0..2: Spatial reference position (centroid of all reference entities)
    if ref_points.shape[0] > 0:
        mean_pos = ref_points.mean(dim=0)
        params_raw[0:3] = mean_pos
        # Causal normalization: tanh(pos / 100.0)
        params[0:3] = torch.tanh(mean_pos / 100.0)
        mask[0:3] = True
    else:
        plane = getattr(workplane, "plane", None)
        if plane is not None:
            try:
                orig = torch.tensor(
                    [plane.origin.x, plane.origin.y, plane.origin.z],
                    dtype=torch.float32,
                )
                params_raw[0:3] = orig
                params[0:3] = torch.tanh(orig / 100.0)
                mask[0:3] = True
            except Exception:
                pass

    # Dims 3..5: Spatial reference direction (mean normalized direction)
    if ref_dirs.shape[0] > 0:
        mean_dir = ref_dirs.mean(dim=0)
        norm = torch.linalg.norm(mean_dir)
        if norm > 1e-6:
            mean_dir = mean_dir / norm
        else:
            mean_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        params_raw[3:6] = mean_dir
        params[3:6] = mean_dir  # Direction is already unit length in [-1, 1]
        mask[3:6] = True
    else:
        plane = getattr(workplane, "plane", None)
        if plane is not None:
            try:
                zdir = torch.tensor(
                    [plane.zDir.x, plane.zDir.y, plane.zDir.z], dtype=torch.float32
                )
                params_raw[3:6] = zdir
                params[3:6] = zdir
                mask[3:6] = True
            except Exception:
                pass

    # Dims 6..11: Operation-specific metric continuous arguments (e.g. distance, radius, depth)
    # SymLog transform: sign(x) * log1p(|x|) preserves dynamic range from 0.1mm to 1000mm
    n_metric = min(len(metric_nums), 6)
    for m_i in range(n_metric):
        val = float(metric_nums[m_i])
        params_raw[6 + m_i] = val
        params[6 + m_i] = float(symlog(torch.tensor(val, dtype=torch.float32)).item())
        mask[6 + m_i] = True

    params_raw = torch.nan_to_num(params_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    params = torch.nan_to_num(params, nan=0.0, posinf=20.0, neginf=-20.0)

    return params_raw, params, mask


def _extract_number(val: Any) -> Optional[float]:
    """Safely extract a float, excluding boolean flags and non-numeric types."""
    if isinstance(val, (int, float, np.number)) and not isinstance(val, (bool, np.bool_)):
        return float(val)
    if isinstance(val, str):
        # Parse only pure numeric strings (e.g. "15.5", "-2.0"), avoiding selector strings like ">Z[1]"
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


def _extract_raw_numeric_args(
    op_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> List[float]:
    """Extract numeric parameters from method call arguments."""
    nums: List[float] = []
    for val in (*args, *kwargs.values()):
        num = _extract_number(val)
        if num is not None:
            nums.append(num)
    return nums


def _extract_pending_profile(workplane: Any) -> Optional[Dict[str, Any]]:
    """Capture 2D sketch wires and profile geometry before consumed by 3D operations."""
    if not hasattr(workplane, "ctx") or not hasattr(
        workplane.ctx, "pendingWires"
    ):
        return None

    wires = list(workplane.ctx.pendingWires)
    if len(wires) == 0:
        return None

    # Classify wires into outer boundary (is_inner=False) vs inner holes/cutouts (is_inner=True)
    is_inner = _classify_pending_wires(wires)
    try:
        outer_idx = is_inner.index(False)
    except ValueError:
        outer_idx = 0

    wire_summaries = []
    total_length = 0.0
    plane = getattr(workplane, "plane", None)

    # Sample 64 points in local 2D workplane coordinates along the true outer boundary wire
    outer_wire = wires[outer_idx]
    points_2d_list = []
    t_vals = np.linspace(0.0, 1.0, 64, endpoint=False)
    for t in t_vals:
        try:
            p_3d = outer_wire.positionAt(float(t))
            if plane is not None:
                p_2d = plane.toLocalCoords(p_3d)
                points_2d_list.append([float(p_2d.x), float(p_2d.y)])
            else:
                points_2d_list.append([float(p_3d.x), float(p_3d.y)])
        except Exception:
            points_2d_list.append([0.0, 0.0])

    for w_idx, w in enumerate(wires):
        try:
            length = float(w.Length())
            bbox = w.BoundingBox()
            center = w.Center()
            total_length += length
            wire_summaries.append(
                {
                    "wire_index": w_idx,
                    "is_inner_loop": bool(is_inner[w_idx]),
                    "length": length,
                    "center": [float(center.x), float(center.y), float(center.z)],
                    "bbox": [
                        float(bbox.xmin),
                        float(bbox.ymin),
                        float(bbox.zmin),
                        float(bbox.xmax),
                        float(bbox.ymax),
                        float(bbox.zmax),
                    ],
                }
            )
        except Exception:
            pass

    wire_primitives = []
    for w_idx, w in enumerate(wires):
        is_inner_loop = bool(is_inner[w_idx])
        try:
            edges = w.Edges() if hasattr(w, "Edges") else []
            for e in edges:
                try:
                    adaptor = BRepAdaptor.BRepAdaptor_Curve(e.wrapped)
                    ctype = adaptor.GetType()
                    p_start = e.startPoint()
                    p_end = e.endPoint()
                    if plane is not None:
                        ps2d = plane.toLocalCoords(p_start)
                        pe2d = plane.toLocalCoords(p_end)
                        start_coord = [float(ps2d.x), float(ps2d.y)]
                        end_coord = [float(pe2d.x), float(pe2d.y)]
                    else:
                        start_coord = [float(p_start.x), float(p_start.y)]
                        end_coord = [float(p_end.x), float(p_end.y)]

                    if ctype == GeomAbs.GeomAbs_CurveType.GeomAbs_Line:
                        type_name = "LINE"
                        radius = 0.0
                    elif ctype == GeomAbs.GeomAbs_CurveType.GeomAbs_Circle:
                        type_name = "CIRCLE" if (hasattr(e, "IsClosed") and e.IsClosed()) else "ARC"
                        circ = adaptor.Circle()
                        radius = float(circ.Radius())
                    else:
                        type_name = "CURVE"
                        radius = 0.0

                    wire_primitives.append({
                        "wire_index": w_idx,
                        "is_inner_loop": is_inner_loop,
                        "type": type_name,
                        "start": start_coord,
                        "end": end_coord,
                        "radius": radius,
                    })
                except Exception:
                    pass
        except Exception:
            pass

    points_2d_tensor = torch.tensor(points_2d_list, dtype=torch.float32)
    points_2d_tensor = torch.nan_to_num(points_2d_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    num_inner = sum(is_inner)
    num_outer = len(wires) - num_inner

    return {
        "num_wires": len(wires),
        "num_outer_loops": num_outer,
        "num_inner_loops": num_inner,
        "total_wire_length": total_length,
        "points_2d": points_2d_tensor,
        "wires": wire_summaries,
        "primitives": wire_primitives,
    }


def _classify_pending_wires(wires: List[Any]) -> List[bool]:
    """Classify wires as outer boundary (False) or inner hole (True)."""
    if len(wires) <= 1:
        return [False] * len(wires)
    try:
        sorted_wires = sorted(
            wires,
            key=lambda w: (w.BoundingBox().xlen * w.BoundingBox().ylen),
            reverse=True,
        )
        face = cq.Face.makeFromWires(sorted_wires[0], sorted_wires[1:])
        outer_occ = ShapeAnalysis.ShapeAnalysis.OuterWire_s(face.wrapped)
        return [not outer_occ.IsSame(w.wrapped) for w in wires]
    except Exception:
        areas = [(w.BoundingBox().xlen * w.BoundingBox().ylen) for w in wires]
        max_idx = int(np.argmax(areas))
        return [i != max_idx for i in range(len(wires))]
