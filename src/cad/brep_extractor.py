"""Boundary Representation (B-Rep) feature extractor for solids and Face-Adjacency Graphs.

Extracts:
  - 32-dim analytical face surface features
  - [N, 7, 16, 16] UV surface grids with valid trimming masks
  - PyTorch Geometric-compatible Face-Adjacency Graph (FAG) with aligned directed edge features
  - Bipartite face-edge incidence index [2, E_inc]
  - Diagnostic geometric validity signatures
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import cadquery as cq
import OCP.Bnd as Bnd
import OCP.BRep as BRep
import OCP.BRepAdaptor as BRepAdaptor
import OCP.BRepBndLib as BRepBndLib
import OCP.BRepGProp as BRepGProp
import OCP.BRepTools as BRepTools
import OCP.BRepTopAdaptor as BRepTopAdaptor
import OCP.GeomAbs as GeomAbs
import OCP.GProp as GProp
import OCP.gp as gp
import OCP.TopAbs as TopAbs
import OCP.TopExp as TopExp
import OCP.TopoDS as TopoDS
import OCP.TopTools as TopTools

from src.data.transforms import symlog
from src.data.schema import EDGE_FEATURE_NAMES, FACE_FEATURE_NAMES, create_empty_state

logger = logging.getLogger(__name__)


def _extract_edges_data(
    edge_map: Any,
    edge_to_faces: Any,
    solid_center: Tuple[float, float, float],
    curve_samples: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract 16-D analytical edge features and sampled 1D curve parameterization."""
    n_edges = edge_map.Size()
    if n_edges == 0:
        return torch.zeros((0, 16), dtype=torch.float32), torch.zeros((0, 6, curve_samples), dtype=torch.float32)

    edge_features_list: List[List[float]] = []
    edges_u_list: List[np.ndarray] = []

    pnt = gp.gp_Pnt()
    t_vec = gp.gp_Vec()

    for e_idx in range(1, n_edges + 1):
        edge_shape = edge_map.FindKey(e_idx)
        edge = TopoDS.TopoDS.Edge(edge_shape)
        adaptor = BRepAdaptor.BRepAdaptor_Curve(edge)
        c_type = adaptor.GetType()

        # 1. 6-dim curve type one-hot
        # [Line, Circle, Ellipse, Parabola/Hyperbola, BSpline/Bezier, Other]
        ct_onehot = [0.0] * 6
        if c_type == GeomAbs.GeomAbs_CurveType.GeomAbs_Line:
            ct_onehot[0] = 1.0
        elif c_type == GeomAbs.GeomAbs_CurveType.GeomAbs_Circle:
            ct_onehot[1] = 1.0
        elif c_type == GeomAbs.GeomAbs_CurveType.GeomAbs_Ellipse:
            ct_onehot[2] = 1.0
        elif c_type in (
            GeomAbs.GeomAbs_CurveType.GeomAbs_Parabola,
            GeomAbs.GeomAbs_CurveType.GeomAbs_Hyperbola,
        ):
            ct_onehot[3] = 1.0
        elif c_type in (
            GeomAbs.GeomAbs_CurveType.GeomAbs_BezierCurve,
            GeomAbs.GeomAbs_CurveType.GeomAbs_BSplineCurve,
        ):
            ct_onehot[4] = 1.0
        else:
            ct_onehot[5] = 1.0

        # 2. Metric arc length (SymLog-scaled)
        c_props = GProp.GProp_GProps()
        try:
            BRepGProp.BRepGProp.LinearProperties_s(edge, c_props)
            raw_len = float(max(0.0, c_props.Mass()))
        except Exception:
            raw_len = 0.0
        scaled_len = float(symlog(torch.tensor(raw_len, dtype=torch.float32)).item())

        # 3. Parametric domain bounds and midpoint
        u_min = adaptor.FirstParameter()
        u_max = adaptor.LastParameter()
        u_mid = 0.5 * (u_min + u_max)

        # Midpoint 3D position and tangent
        try:
            adaptor.D1(u_mid, pnt, t_vec)
            midpoint = [float(pnt.X()), float(pnt.Y()), float(pnt.Z())]
            mag = t_vec.Magnitude()
            if mag > 1e-6:
                t_vec.Normalize()
                tangent = [float(t_vec.X()), float(t_vec.Y()), float(t_vec.Z())]
            else:
                tangent = [0.0, 0.0, 1.0]
        except Exception:
            midpoint = [0.0, 0.0, 0.0]
            tangent = [0.0, 0.0, 1.0]

        # Normalized midpoint: tanh(coord / 100.0)
        mid_norm = [float(np.tanh(m / 100.0)) for m in midpoint]

        # 4. Topological flags
        is_closed = 1.0 if (adaptor.IsClosed() or edge.Closed()) else 0.0
        try:
            is_degen = 1.0 if BRep.BRep_Tool.Degenerated_s(edge) else 0.0
        except Exception:
            is_degen = 0.0

        # 5. Convexity between incident faces
        # +1.0 for convex, -1.0 for concave, 0.0 for smooth or laminar boundary
        convexity = 0.0
        e_adj_idx = edge_to_faces.FindIndex(edge)
        if e_adj_idx > 0:
            flist = edge_to_faces.FindFromIndex(e_adj_idx)
            if flist.Size() == 2:
                try:
                    f1 = cq.Face(flist.First())
                    f2 = cq.Face(flist.Last())
                    cq_pnt = cq.Vector(midpoint[0], midpoint[1], midpoint[2])
                    n1 = f1.normalAt(cq_pnt)
                    n2 = f2.normalAt(cq_pnt)
                    if n1.dot(n2) > 0.999:
                        convexity = 0.0
                    else:
                        v_solid = cq.Vector(
                            midpoint[0] - solid_center[0],
                            midpoint[1] - solid_center[1],
                            midpoint[2] - solid_center[2],
                        )
                        n_sum = n1 + n2
                        convexity = 1.0 if (n_sum.dot(v_solid) > 0) else -1.0
                except Exception:
                    convexity = 0.0

        # Assemble 16-D feature vector
        feat_16 = ct_onehot + [scaled_len] + mid_norm + tangent + [is_closed, is_degen, convexity]
        edge_features_list.append(feat_16)

        # 6. Sample 1D curve: edges_u [6, curve_samples]
        # Channels [x, y, z, tx, ty, tz] across equidistant parameter samples
        u_samples = np.linspace(u_min, u_max, curve_samples)
        u_grid = np.zeros((6, curve_samples), dtype=np.float32)
        for s_idx, u_val in enumerate(u_samples):
            try:
                adaptor.D1(float(u_val), pnt, t_vec)
                u_grid[0, s_idx] = pnt.X()
                u_grid[1, s_idx] = pnt.Y()
                u_grid[2, s_idx] = pnt.Z()
                mag = t_vec.Magnitude()
                if mag > 1e-6:
                    t_vec.Normalize()
                    u_grid[3, s_idx] = t_vec.X()
                    u_grid[4, s_idx] = t_vec.Y()
                    u_grid[5, s_idx] = t_vec.Z()
                else:
                    u_grid[3, s_idx] = tangent[0]
                    u_grid[4, s_idx] = tangent[1]
                    u_grid[5, s_idx] = tangent[2]
            except Exception:
                u_grid[0:3, s_idx] = midpoint
                u_grid[3:6, s_idx] = tangent

        edges_u_list.append(u_grid)

    edges_features = torch.tensor(edge_features_list, dtype=torch.float32)
    edges_features = torch.nan_to_num(edges_features, nan=0.0, posinf=1.0, neginf=-1.0)

    edges_u = torch.tensor(np.array(edges_u_list), dtype=torch.float32)
    edges_u = torch.nan_to_num(edges_u, nan=0.0, posinf=1.0, neginf=-1.0)

    return edges_features, edges_u


def extract_brep_state(
    shape_obj: Any,
    images_tensor: Optional[torch.Tensor] = None,
    uv_grid_size: int = 16,
    curve_samples: int = 16,
) -> Dict[str, Any]:
    """Extract complete B-Rep state from a CadQuery Workplane/Shape or TopoDS_Shape.

    Args:
        shape_obj: Workplane, Compound, Solid, or TopoDS_Shape.
        images_tensor: In-memory rendered [4, 3, 224, 224] uint8 multi-view tensor.
        uv_grid_size: Resolution of 2D UV grid sampled on face surfaces (default: 16).
        curve_samples: Number of 1D curve parameter points sampled on edges (default: 16).

    Returns:
        state: Validated State dictionary.
    """
    if shape_obj is None:
        empty = create_empty_state()
        if images_tensor is not None:
            empty["images"] = images_tensor
        return empty

    # Safely extract TopoDS_Shape supporting Single Solid or Compounds
    if hasattr(shape_obj, "toCompound"):
        occ_shape = shape_obj.toCompound().wrapped
    elif hasattr(shape_obj, "val"):
        val_obj = shape_obj.val()
        if val_obj is None:
            empty = create_empty_state()
            if images_tensor is not None:
                empty["images"] = images_tensor
            return empty
        occ_shape = val_obj.wrapped if hasattr(val_obj, "wrapped") else val_obj
    elif hasattr(shape_obj, "wrapped"):
        occ_shape = shape_obj.wrapped
    else:
        occ_shape = shape_obj

    if occ_shape is None or (hasattr(occ_shape, "IsNull") and occ_shape.IsNull()):
        empty = create_empty_state()
        if images_tensor is not None:
            empty["images"] = images_tensor
        return empty

    # Map all faces deterministically
    face_map = TopTools.TopTools_IndexedMapOfShape()
    TopExp.TopExp.MapShapes_s(occ_shape, TopAbs.TopAbs_ShapeEnum.TopAbs_FACE, face_map)
    n_faces = face_map.Size()

    if n_faces == 0:
        empty = create_empty_state()
        if images_tensor is not None:
            empty["images"] = images_tensor
        return empty

    # Map all topological edges deterministically
    edge_map = TopTools.TopTools_IndexedMapOfShape()
    TopExp.TopExp.MapShapes_s(occ_shape, TopAbs.TopAbs_ShapeEnum.TopAbs_EDGE, edge_map)
    n_edges = edge_map.Size()

    # Map edges to adjacent faces
    edge_to_faces = TopTools.TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.TopExp.MapShapesAndAncestors_s(
        occ_shape,
        TopAbs.TopAbs_ShapeEnum.TopAbs_EDGE,
        TopAbs.TopAbs_ShapeEnum.TopAbs_FACE,
        edge_to_faces,
    )

    # Solid Volume & Center of Mass
    vol_props = GProp.GProp_GProps()
    try:
        BRepGProp.BRepGProp.VolumeProperties_s(occ_shape, vol_props)
        volume = float(max(0.0, vol_props.Mass()))
        com_s = vol_props.CentreOfMass()
        solid_center = (float(com_s.X()), float(com_s.Y()), float(com_s.Z()))
    except Exception:
        volume = 0.0
        solid_center = (0.0, 0.0, 0.0)

    # Extract 16-D edges_features and [6, 16] edges_u
    edges_features, edges_u = _extract_edges_data(
        edge_map=edge_map,
        edge_to_faces=edge_to_faces,
        solid_center=solid_center,
        curve_samples=curve_samples,
    )

    # Build Face-Adjacency Graph (FAG) from shared boundary edges preserving physical edge indices
    directed_edges: List[Tuple[int, int, int]] = []
    for i in range(1, edge_to_faces.Size() + 1):
        flist = edge_to_faces.FindFromIndex(i)
        edge_shape = edge_to_faces.FindKey(i)
        phys_edge_idx = edge_map.FindIndex(edge_shape) - 1
        if phys_edge_idx < 0 or phys_edge_idx >= n_edges:
            continue

        shared = [face_map.FindIndex(f) - 1 for f in flist]
        shared = [idx for idx in shared if 0 <= idx < n_faces]
        if len(shared) == 2 and shared[0] != shared[1]:
            u, v = shared[0], shared[1]
            directed_edges.append((u, v, phys_edge_idx))
            directed_edges.append((v, u, phys_edge_idx))
        elif len(shared) > 2:
            unique_shared = sorted(list(set(shared)))
            for a in range(len(unique_shared)):
                for b in range(a + 1, len(unique_shared)):
                    directed_edges.append((unique_shared[a], unique_shared[b], phys_edge_idx))
                    directed_edges.append((unique_shared[b], unique_shared[a], phys_edge_idx))

    if directed_edges:
        unique_directed = sorted(list(set(directed_edges)))
        src = [e[0] for e in unique_directed]
        dst = [e[1] for e in unique_directed]
        edge_ptrs = [e[2] for e in unique_directed]

        faces_adjacency_index = torch.tensor([src, dst], dtype=torch.long)
        faces_adjacency_edge_indices = torch.tensor(edge_ptrs, dtype=torch.long)
        edges_features_directed = edges_features[faces_adjacency_edge_indices]

        # Reverse edge indices for UV-Net style edge fusion
        edge_to_idx = {(e[0], e[1], e[2]): idx for idx, e in enumerate(unique_directed)}
        rev_indices = [edge_to_idx.get((e[1], e[0], e[2]), idx) for idx, e in enumerate(unique_directed)]
        reverse_edge_indices = torch.tensor(rev_indices, dtype=torch.long)
    else:
        faces_adjacency_index = torch.zeros((2, 0), dtype=torch.long)
        faces_adjacency_edge_indices = torch.zeros(0, dtype=torch.long)
        edges_features_directed = torch.zeros((0, 16), dtype=torch.float32)
        reverse_edge_indices = torch.zeros(0, dtype=torch.long)

    # Bipartite face-edge incidence index [2, E_inc]
    face_edge_list: List[Tuple[int, int]] = []
    f_edge_map = TopTools.TopTools_IndexedMapOfShape()
    for f_idx in range(1, n_faces + 1):
        face_shape = face_map.FindKey(f_idx)
        f_edge_map.Clear()
        TopExp.TopExp.MapShapes_s(face_shape, TopAbs.TopAbs_ShapeEnum.TopAbs_EDGE, f_edge_map)
        for fe_i in range(1, f_edge_map.Size() + 1):
            e_shape = f_edge_map.FindKey(fe_i)
            e_idx = edge_map.FindIndex(e_shape) - 1
            if 0 <= e_idx < n_edges:
                face_edge_list.append((f_idx - 1, e_idx))

    if face_edge_list:
        unique_fe = sorted(list(set(face_edge_list)))
        face_edge_index = torch.tensor(unique_fe, dtype=torch.long).t().contiguous()
    else:
        face_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Count number of individual solids
    solid_map = TopTools.TopTools_IndexedMapOfShape()
    TopExp.TopExp.MapShapes_s(occ_shape, TopAbs.TopAbs_ShapeEnum.TopAbs_SOLID, solid_map)
    num_solids = solid_map.Size()

    # Extract 32-dim features and 7x16x16 UV grids
    node_features: List[List[float]] = []
    uv_tensors: List[np.ndarray] = []
    total_surface_area = 0.0

    pnt = gp.gp_Pnt()
    d1u = gp.gp_Vec()
    d1v = gp.gp_Vec()

    for f_idx in range(1, n_faces + 1):
        face_shape = face_map.FindKey(f_idx)
        face = TopoDS.TopoDS.Face(face_shape)
        adaptor = BRepAdaptor.BRepAdaptor_Surface(face)
        st = adaptor.GetType()

        # 1. 8-dim surface type one-hot
        st_onehot = [0.0] * 8
        if st == GeomAbs.GeomAbs_SurfaceType.GeomAbs_Plane:
            st_onehot[0] = 1.0
        elif st == GeomAbs.GeomAbs_SurfaceType.GeomAbs_Cylinder:
            st_onehot[1] = 1.0
        elif st == GeomAbs.GeomAbs_SurfaceType.GeomAbs_Cone:
            st_onehot[2] = 1.0
        elif st == GeomAbs.GeomAbs_SurfaceType.GeomAbs_Sphere:
            st_onehot[3] = 1.0
        elif st == GeomAbs.GeomAbs_SurfaceType.GeomAbs_Torus:
            st_onehot[4] = 1.0
        elif st in (
            GeomAbs.GeomAbs_SurfaceType.GeomAbs_BezierSurface,
            GeomAbs.GeomAbs_SurfaceType.GeomAbs_BSplineSurface,
        ):
            st_onehot[5] = 1.0
        elif st in (
            GeomAbs.GeomAbs_SurfaceType.GeomAbs_SurfaceOfRevolution,
            GeomAbs.GeomAbs_SurfaceType.GeomAbs_SurfaceOfExtrusion,
        ):
            st_onehot[6] = 1.0
        else:
            st_onehot[7] = 1.0

        # 2. Area and Center of Mass
        props = GProp.GProp_GProps()
        BRepGProp.BRepGProp.SurfaceProperties_s(face, props)
        area = float(props.Mass())
        total_surface_area += area
        com = props.CentreOfMass()
        centroid = [float(com.X()), float(com.Y()), float(com.Z())]

        # 3. Parametric domain bounds using BRepTools.UVBounds_s (avoids infinite bounds)
        umin, umax, vmin, vmax = BRepTools.BRepTools.UVBounds_s(face)
        u_mid = 0.5 * (umin + umax)
        v_mid = 0.5 * (vmin + vmax)

        # 4. Representative normal vector at midpoint UV
        try:
            adaptor.D1(u_mid, v_mid, pnt, d1u, d1v)
            norm_vec = d1u.Crossed(d1v)
            if norm_vec.Magnitude() > 1e-6:
                norm_vec.Normalize()
                normal = [float(norm_vec.X()), float(norm_vec.Y()), float(norm_vec.Z())]
            else:
                normal = [0.0, 0.0, 1.0]
        except Exception:
            normal = [0.0, 0.0, 1.0]

        # Correct OCC face orientation
        is_reversed = face.Orientation() == TopAbs.TopAbs_Orientation.TopAbs_REVERSED
        if is_reversed:
            normal = [-n for n in normal]

        # 5. Bounding box extents
        bbox_face = Bnd.Bnd_Box()
        BRepBndLib.BRepBndLib.Add_s(face, bbox_face)
        f_xmin, f_ymin, f_zmin, f_xmax, f_ymax, f_zmax = bbox_face.Get()
        bbox_dims = [float(f_xmax - f_xmin), float(f_ymax - f_ymin), float(f_zmax - f_zmin)]

        # 6. Topological counts
        wires = TopTools.TopTools_IndexedMapOfShape()
        TopExp.TopExp.MapShapes_s(face, TopAbs.TopAbs_ShapeEnum.TopAbs_WIRE, wires)
        face_edges = TopTools.TopTools_IndexedMapOfShape()
        TopExp.TopExp.MapShapes_s(face, TopAbs.TopAbs_ShapeEnum.TopAbs_EDGE, face_edges)
        topo_counts = [float(wires.Size()), float(face_edges.Size())]

        # 7. Orientation flag
        orientation = [-1.0 if is_reversed else 1.0]

        # 8. UV domain spans
        uv_params = [float(umax - umin), float(vmax - vmin)]

        # 9. Matrix of inertia diagonal
        inertia = props.MatrixOfInertia()
        diag_inertia = [float(inertia.Value(1, 1)), float(inertia.Value(2, 2)), float(inertia.Value(3, 3))]

        # 10. Invariant derived descriptors
        log_area = [float(np.log1p(max(0.0, area)))]
        aspect = [
            float(bbox_dims[0] / (bbox_dims[1] + 1e-5)),
            float(bbox_dims[1] / (bbox_dims[2] + 1e-5)),
        ]
        diag_len = [float(np.sqrt(sum(d**2 for d in bbox_dims)))]
        curv_flag = [0.0 if st == GeomAbs.GeomAbs_SurfaceType.GeomAbs_Plane else 1.0]
        extra_flag = [1.0 if wires.Size() > 1 else 0.0]

        feat_vector = (
            st_onehot
            + [area]
            + centroid
            + normal
            + bbox_dims
            + topo_counts
            + orientation
            + uv_params
            + diag_inertia
            + log_area
            + aspect
            + diag_len
            + curv_flag
            + extra_flag
        )
        assert len(feat_vector) == 32
        node_features.append(feat_vector)

        # Extract 7xNxN UV surface tensor with FaceClassifier trimming mask
        classifier = BRepTopAdaptor.BRepTopAdaptor_FClass2d(face, 1e-6)
        u_grid = np.linspace(umin, umax, uv_grid_size)
        v_grid = np.linspace(vmin, vmax, uv_grid_size)
        uv_tensor = np.zeros((7, uv_grid_size, uv_grid_size), dtype=np.float32)

        for ui, u in enumerate(u_grid):
            for vi, v in enumerate(v_grid):
                p2d = gp.gp_Pnt2d(float(u), float(v))
                st_2d = classifier.Perform(p2d)
                if st_2d in (
                    TopAbs.TopAbs_State.TopAbs_IN,
                    TopAbs.TopAbs_State.TopAbs_ON,
                ):
                    uv_tensor[6, ui, vi] = 1.0  # valid trimming mask
                    try:
                        adaptor.D1(float(u), float(v), pnt, d1u, d1v)
                        n_vec = d1u.Crossed(d1v)
                        if n_vec.Magnitude() > 1e-6:
                            n_vec.Normalize()
                            nx, ny, nz = float(n_vec.X()), float(n_vec.Y()), float(n_vec.Z())
                        else:
                            nx, ny, nz = normal[0], normal[1], normal[2]
                    except Exception:
                        # Singularity safeguard: jitter towards domain center
                        u_j = float(u) + 1e-4 * (u_mid - float(u))
                        v_j = float(v) + 1e-4 * (v_mid - float(v))
                        try:
                            adaptor.D1(u_j, v_j, pnt, d1u, d1v)
                            n_vec = d1u.Crossed(d1v)
                            if n_vec.Magnitude() > 1e-6:
                                n_vec.Normalize()
                                nx, ny, nz = float(n_vec.X()), float(n_vec.Y()), float(n_vec.Z())
                            else:
                                nx, ny, nz = normal[0], normal[1], normal[2]
                        except Exception:
                            nx, ny, nz = normal[0], normal[1], normal[2]

                    if is_reversed:
                        nx, ny, nz = -nx, -ny, -nz

                    uv_tensor[0, ui, vi] = float(pnt.X())
                    uv_tensor[1, ui, vi] = float(pnt.Y())
                    uv_tensor[2, ui, vi] = float(pnt.Z())
                    uv_tensor[3, ui, vi] = nx
                    uv_tensor[4, ui, vi] = ny
                    uv_tensor[5, ui, vi] = nz

        uv_tensors.append(uv_tensor)

    # Convert faces to sanitized PyTorch tensors
    faces_features = torch.tensor(node_features, dtype=torch.float32)
    faces_features = torch.nan_to_num(faces_features, nan=0.0, posinf=1.0, neginf=-1.0)

    faces_uv = torch.tensor(np.array(uv_tensors), dtype=torch.float32)
    faces_uv = torch.nan_to_num(faces_uv, nan=0.0, posinf=1.0, neginf=-1.0)

    # Solid Bounding Box
    bbox_shape = Bnd.Bnd_Box()
    BRepBndLib.BRepBndLib.Add_s(occ_shape, bbox_shape)
    s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax = bbox_shape.Get()
    bbox_tensor = torch.tensor(
        [float(s_xmin), float(s_ymin), float(s_zmin), float(s_xmax), float(s_ymax), float(s_zmax)],
        dtype=torch.float32,
    )
    bbox_tensor = torch.nan_to_num(bbox_tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    # Images
    if images_tensor is None:
        images_tensor = torch.full((4, 3, 224, 224), 220, dtype=torch.uint8)

    return {
        "faces_features": faces_features,
        "faces_uv": faces_uv,
        "edges_features": edges_features,
        "edges_features_directed": edges_features_directed,
        "edges_u": edges_u,
        "faces_adjacency_index": faces_adjacency_index,
        "faces_adjacency_edge_indices": faces_adjacency_edge_indices,
        "reverse_edge_indices": reverse_edge_indices,
        "face_edge_index": face_edge_index,
        "images": images_tensor,
        "num_faces": n_faces,
        "num_edges": n_edges,
        "num_solids": num_solids,
        "volume": volume,
        "surface_area": total_surface_area,
        "bbox": bbox_tensor,
    }
