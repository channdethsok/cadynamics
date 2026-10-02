"""Data schemas, feature definitions, and validation helpers for CADynamics.

Ensures mathematical and topological integrity across states, actions,
trajectories, and on-disk shard files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch

from src.data.vocabulary import (
    CANONICAL_COMMANDS,
    NUM_COMMANDS,
    REF_KIND_TO_NAME,
    VOCABULARY_VERSION,
)

SCHEMA_VERSION: str = "0.3.0"

# Canonical 4-view camera perspectives for multiview CAD rendering
CANONICAL_VIEW_NAMES: Tuple[str, ...] = ("iso", "front", "top", "right")

# Explicit 32 scalar features per B-Rep face
FACE_FEATURE_NAMES: Tuple[str, ...] = (
    # [0..7]: 8-dim surface type one-hot encoding
    "is_plane",                    # 0: Planar surface
    "is_cylinder",                 # 1: Cylindrical surface
    "is_cone",                     # 2: Conical surface
    "is_sphere",                   # 3: Spherical surface
    "is_torus",                    # 4: Toroidal surface
    "is_bspline",                  # 5: B-spline / Bezier freeform surface
    "is_revolution_or_extrusion",  # 6: Surface of revolution or extrusion
    "is_other_surface",            # 7: Offset or other analytical surface
    # [8]: Metric area
    "area",                        # 8: Surface area
    # [9..11]: Centroid / Center of Mass
    "centroid_x",                  # 9: Centroid X
    "centroid_y",                  # 10: Centroid Y
    "centroid_z",                  # 11: Centroid Z
    # [12..14]: Representative oriented normal vector
    "normal_x",                    # 12: Normal X at mid UV parameter
    "normal_y",                    # 13: Normal Y at mid UV parameter
    "normal_z",                    # 14: Normal Z at mid UV parameter
    # [15..17]: Oriented bounding box dimensions
    "bbox_dx",                     # 15: Face bounding box delta X
    "bbox_dy",                     # 16: Face bounding box delta Y
    "bbox_dz",                     # 17: Face bounding box delta Z
    # [18..19]: Topological complexity
    "num_wires",                   # 18: Number of trimming boundary loops (wires)
    "num_edges",                   # 19: Number of boundary edges
    # [20]: Orientation flag
    "orientation",                 # 20: +1.0 for TopAbs_FORWARD, -1.0 for TopAbs_REVERSED
    # [21..22]: UV parameter domain spans
    "u_range_len",                 # 21: u_max - u_min
    "v_range_len",                 # 22: v_max - v_min
    # [23..25]: Diagonal of matrix of inertia
    "inertia_ixx",                 # 23: Inertia Ixx
    "inertia_iyy",                 # 24: Inertia Iyy
    "inertia_izz",                 # 25: Inertia Izz
    # [26..31]: Invariant geometric descriptors
    "log_area",                    # 26: log(1 + area)
    "aspect_xy",                   # 27: bbox_dx / (bbox_dy + eps)
    "aspect_yz",                   # 28: bbox_dy / (bbox_dz + eps)
    "bbox_diagonal",               # 29: sqrt(dx^2 + dy^2 + dz^2)
    "is_curved",                   # 30: 0.0 if plane, 1.0 if curved
    "is_multi_loop",               # 31: 1.0 if wires > 1, 0.0 otherwise
)

assert len(FACE_FEATURE_NAMES) == 32, f"Expected exactly 32 face features, got {len(FACE_FEATURE_NAMES)}"

# Explicit 16 scalar features per B-Rep edge
EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    # [0..5]: 6-dim curve type one-hot encoding
    "is_line",                     # 0: Linear edge
    "is_circle",                   # 1: Circular arc or circle
    "is_ellipse",                  # 2: Elliptical arc or ellipse
    "is_parabola_hyperbola",       # 3: Parabolic or hyperbolic curve
    "is_bspline_bezier",           # 4: Freeform B-spline or Bezier curve
    "is_other_curve",              # 5: Offset or other curve type
    # [6]: Curve length
    "length",                      # 6: Metric arc length (SymLog-scaled)
    # [7..9]: Curve Midpoint
    "midpoint_x",                  # 7: Midpoint X (tanh normalized)
    "midpoint_y",                  # 8: Midpoint Y (tanh normalized)
    "midpoint_z",                  # 9: Midpoint Z (tanh normalized)
    # [10..12]: Normalized tangent at midpoint
    "tangent_x",                   # 10: Unit tangent X at t=0.5
    "tangent_y",                   # 11: Unit tangent Y at t=0.5
    "tangent_z",                   # 12: Unit tangent Z at t=0.5
    # [13..15]: Topological & geometric flags
    "is_closed",                   # 13: 1.0 if closed curve loop, 0.0 otherwise
    "is_degenerated",              # 14: 1.0 if degenerated to a point, 0.0 otherwise
    "convexity",                   # 15: +1.0 for convex edge, -1.0 for concave, 0.0 for smooth/laminar
)

assert len(EDGE_FEATURE_NAMES) == 16, f"Expected exactly 16 edge features, got {len(EDGE_FEATURE_NAMES)}"


def create_empty_state() -> Dict[str, Any]:
    """Construct deterministic empty workspace state S_0 with 4 canonical views."""
    neutral_canvas = torch.full((4, 3, 224, 224), 220, dtype=torch.uint8)
    return {
        "faces_features": torch.zeros((0, 32), dtype=torch.float32),
        "faces_uv": torch.zeros((0, 7, 16, 16), dtype=torch.float32),
        "edges_features": torch.zeros((0, 16), dtype=torch.float32),
        "edges_u": torch.zeros((0, 6, 16), dtype=torch.float32),
        "faces_adjacency_index": torch.zeros((2, 0), dtype=torch.long),
        "faces_adjacency_edge_indices": torch.zeros(0, dtype=torch.long),
        "edges_features_directed": torch.zeros((0, 16), dtype=torch.float32),
        "face_edge_index": torch.zeros((2, 0), dtype=torch.long),
        "reverse_edge_indices": torch.zeros(0, dtype=torch.long),
        "images": neutral_canvas,
        "num_faces": 0,
        "num_edges": 0,
        "num_solids": 0,
        "volume": 0.0,
        "surface_area": 0.0,
        "bbox": torch.zeros(6, dtype=torch.float32),
    }


def validate_state(state: Dict[str, Any], is_empty: bool = False) -> None:
    """Validate that state dictionary strictly conforms to invariants."""
    required_keys = [
        "faces_features",
        "faces_uv",
        "edges_features",
        "edges_u",
        "faces_adjacency_index",
        "images",
        "num_faces",
        "num_edges",
        "num_solids",
        "volume",
        "surface_area",
        "bbox",
    ]
    for k in required_keys:
        if k not in state:
            raise ValueError(f"State missing required key: '{k}'")

    N_faces = state["num_faces"]
    N_edges = state["num_edges"]
    f_nodes = state["faces_features"]
    f_uv = state["faces_uv"]
    e_feats = state["edges_features"]
    e_u = state["edges_u"]
    adj_edges = state["faces_adjacency_index"]
    imgs = state["images"]

    if f_nodes.ndim != 2 or f_nodes.shape != (N_faces, 32):
        raise ValueError(f"faces_features shape mismatch: expected ({N_faces}, 32), got {f_nodes.shape}")
    if f_uv.ndim != 4 or f_uv.shape != (N_faces, 7, 16, 16):
        raise ValueError(f"faces_uv shape mismatch: expected ({N_faces}, 7, 16, 16), got {f_uv.shape}")
    if e_feats.ndim != 2 or e_feats.shape != (N_edges, 16):
        raise ValueError(f"edges_features shape mismatch: expected ({N_edges}, 16), got {e_feats.shape}")
    if e_u.ndim != 3 or e_u.shape != (N_edges, 6, 16):
        raise ValueError(f"edges_u shape mismatch: expected ({N_edges}, 6, 16), got {e_u.shape}")
    if adj_edges.ndim != 2 or adj_edges.shape[0] != 2:
        raise ValueError(f"faces_adjacency_index shape mismatch: expected (2, E), got {adj_edges.shape}")
    if imgs.ndim != 4 or imgs.shape != (4, 3, 224, 224) or imgs.dtype != torch.uint8:
        raise ValueError(f"images shape or dtype mismatch: expected uint8 (4, 3, 224, 224), got {imgs.dtype} {imgs.shape}")

    # Check finite
    if not torch.all(torch.isfinite(f_nodes)):
        raise ValueError("faces_features contains non-finite values (NaN/Inf)")
    if not torch.all(torch.isfinite(f_uv)):
        raise ValueError("faces_uv contains non-finite values (NaN/Inf)")
    if not torch.all(torch.isfinite(e_feats)):
        raise ValueError("edges_features contains non-finite values (NaN/Inf)")
    if not torch.all(torch.isfinite(e_u)):
        raise ValueError("edges_u contains non-finite values (NaN/Inf)")

    # Check UV mask validity (channel 6 must be 0.0 or 1.0)
    if N_faces > 0:
        mask = f_uv[:, 6, :, :]
        if not torch.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("faces_uv trimming mask contains non-binary values")

    # Check FAG index validity
    E_adj = adj_edges.shape[1]
    if E_adj > 0:
        if adj_edges.min().item() < 0 or adj_edges.max().item() >= N_faces:
            raise ValueError(f"faces_adjacency_index contains out-of-bounds face index [0, {N_faces-1}]: [{adj_edges.min()}, {adj_edges.max()}]")

    # Optional topological completeness checks (Schema 0.4.0)
    if "faces_adjacency_edge_indices" in state:
        f_adj_e = state["faces_adjacency_edge_indices"]
        if f_adj_e.ndim != 1 or f_adj_e.shape[0] != E_adj:
            raise ValueError(f"faces_adjacency_edge_indices length {f_adj_e.shape} != E_adj {E_adj}")
        if E_adj > 0 and N_edges > 0:
            if f_adj_e.min().item() < 0 or f_adj_e.max().item() >= N_edges:
                raise ValueError("faces_adjacency_edge_indices contains out-of-bounds edge index")

    if "edges_features_directed" in state:
        e_dir = state["edges_features_directed"]
        if e_dir.ndim != 2 or e_dir.shape != (E_adj, 16):
            raise ValueError(f"edges_features_directed shape mismatch: expected ({E_adj}, 16), got {e_dir.shape}")
        if not torch.all(torch.isfinite(e_dir)):
            raise ValueError("edges_features_directed contains non-finite values")

    if "face_edge_index" in state:
        fe_idx = state["face_edge_index"]
        if fe_idx.ndim != 2 or fe_idx.shape[0] != 2:
            raise ValueError(f"face_edge_index shape mismatch: expected (2, E_inc), got {fe_idx.shape}")
        if fe_idx.shape[1] > 0 and N_faces > 0 and N_edges > 0:
            if fe_idx[0].min().item() < 0 or fe_idx[0].max().item() >= N_faces:
                raise ValueError("face_edge_index contains out-of-bounds face index")
            if fe_idx[1].min().item() < 0 or fe_idx[1].max().item() >= N_edges:
                raise ValueError("face_edge_index contains out-of-bounds edge index")

    if "reverse_edge_indices" in state:
        rev_e = state["reverse_edge_indices"]
        if rev_e.ndim != 1 or rev_e.shape[0] != E_adj:
            raise ValueError(f"reverse_edge_indices length {rev_e.shape} != E_adj {E_adj}")


def validate_action(action: Dict[str, Any]) -> None:
    """Validate action structure and invariant constraints."""
    required_keys = [
        "cmd_id",
        "params_raw",
        "params",
        "param_mask",
        "ref_kind",
        "ref_entity_indices",
        "ref_points",
        "ref_directions",
        "operation_name",
    ]
    for k in required_keys:
        if k not in action:
            raise ValueError(f"Action missing required key: '{k}'")

    cmd_id = action["cmd_id"]
    if not isinstance(cmd_id, int) or cmd_id < 0 or cmd_id >= NUM_COMMANDS:
        raise ValueError(f"Invalid action cmd_id: {cmd_id} (must be in [0, {NUM_COMMANDS - 1}])")

    params_raw = action["params_raw"]
    params = action["params"]
    mask = action["param_mask"]

    if params_raw.shape != (12,):
        raise ValueError(f"params_raw shape mismatch: {params_raw.shape}")
    if params.shape != (12,):
        raise ValueError(f"params shape mismatch: {params.shape}")
    if mask.shape != (12,) or mask.dtype != torch.bool:
        raise ValueError(f"param_mask shape/dtype mismatch: {mask.shape}, {mask.dtype}")

    if not torch.all(torch.isfinite(params)):
        raise ValueError("action params contains non-finite values")

    ref_indices = action["ref_entity_indices"]
    ref_points = action["ref_points"]
    ref_dirs = action["ref_directions"]

    M = ref_indices.shape[0]
    if ref_points.shape != (M, 3) or ref_dirs.shape != (M, 3):
        raise ValueError(f"Reference arrays length mismatch: indices={M}, points={ref_points.shape}, dirs={ref_dirs.shape}")


def validate_trajectory(trajectory: Dict[str, Any]) -> None:
    """Validate that trajectory preserves causal invariant len(states) == len(actions) + 1."""
    required_keys = ["part_id", "source_split", "source_shard", "source_row", "num_steps", "states", "actions"]
    for k in required_keys:
        if k not in trajectory:
            raise ValueError(f"Trajectory missing key: '{k}'")

    states = trajectory["states"]
    actions = trajectory["actions"]
    num_steps = trajectory["num_steps"]

    if len(states) != len(actions) + 1:
        raise ValueError(f"Trajectory causal invariant violated: len(states)={len(states)} != len(actions)+1={len(actions)+1}")
    if num_steps != len(actions):
        raise ValueError(f"num_steps={num_steps} != len(actions)={len(actions)}")

    validate_state(states[0], is_empty=True)
    for s in states[1:]:
        validate_state(s, is_empty=False)
    for a in actions:
        validate_action(a)


def validate_shard(shard_data: Dict[str, Any]) -> None:
    """Validate complete shard container."""
    if "schema_version" not in shard_data or shard_data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Shard schema version mismatch: {shard_data.get('schema_version')} != {SCHEMA_VERSION}")
    if "metadata" not in shard_data or "records" not in shard_data:
        raise ValueError("Shard missing 'metadata' or 'records'")
    for rec in shard_data["records"]:
        validate_trajectory(rec)
