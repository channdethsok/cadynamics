"""End-to-end preprocessing test for CADynamics.

Verifies:
  1. Raw Parquet row ingestion from Zero-to-CAD dataset.
  2. AST execution and solid tracking via CadQueryRuntimeTracer.
  3. B-Rep topological extraction (faces, edges, UV grid, 1D curve samples).
  4. Offscreen 4-view canonical rendering (224x224 RGB).
  5. Action grounding (parameters, masks, 2D sketch wires).
  6. Schema 0.3.0 trajectory validation (S_{k+1} = F(S_k, A_k)).
  7. Shard .pt serialization and validate_shard().
  8. DataLoader transition loading and batch collation from the fresh shard.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import tempfile

import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

cq = pytest.importorskip("cadquery", reason="CadQuery is required for preprocessing test")
pv = pytest.importorskip("pyvista", reason="PyVista is required for preprocessing test")

# Off-screen PyVista context
try:
    _plotter = pv.Plotter(off_screen=True, window_size=[10, 10])
    _plotter.render()
    _plotter.close()
except Exception:
    pass

from src.cad.tracer import CadQueryRuntimeTracer
from src.cad.renderer import HeadlessCadRenderer
from src.data.dataset import CADTransitionDataset, collate_transition_batch
from src.data.schema import (
    EDGE_FEATURE_NAMES,
    FACE_FEATURE_NAMES,
    SCHEMA_VERSION,
    validate_shard,
    validate_trajectory,
)
from src.data.vocabulary import CMD2ID, NUM_COMMANDS, REF_KIND_TO_NAME

RAW_VAL_PARQUET = Path("data/zero_to_cad_100k/data/val/85_5c8bc6f0459e4abba11cfaec2884d1f8_000000_000000-0.parquet")


def test_e2e_preprocessing_and_dataloader(tmp_path: Path) -> None:
    """Test full pipeline from raw Parquet row to batched neural tensors."""
    assert RAW_VAL_PARQUET.exists(), f"Raw parquet dataset shard not found at {RAW_VAL_PARQUET}"

    # 1. Read first 2 parts from raw Parquet
    table = pq.read_table(str(RAW_VAL_PARQUET), columns=["uuid", "cadquery_file", "cadquery_ops_count"])
    rows = table.to_pylist()[:2]
    assert len(rows) == 2, "Expected at least 2 parts in raw Parquet shard"

    renderer = HeadlessCadRenderer(image_size=224)
    trajectories = []

    for r_idx, row in enumerate(rows):
        uuid_str = row["uuid"]
        raw_code = row["cadquery_file"]
        code_str = raw_code.decode("utf-8") if isinstance(raw_code, bytes) else str(raw_code)

        # Pre-pass: evaluate bounding box for unified camera framing
        compiled_code = compile(code_str, f"<test_cad_{uuid_str}>", "exec")
        pre_env = {"cq": cq, "cadquery": cq}
        exec(compiled_code, pre_env)
        target = pre_env.get("result") or pre_env.get("part")
        if target is None:
            for val in pre_env.values():
                if isinstance(val, cq.Workplane) and val.val() is not None:
                    target = val
                    break
        assert target is not None and isinstance(target, cq.Workplane), f"No valid Workplane found in {uuid_str}"

        bb = target.val().BoundingBox()
        reference_bbox = [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax]

        # Trace execution with CadQueryRuntimeTracer
        tracer = CadQueryRuntimeTracer(
            render_images=True,
            renderer=renderer,
            reference_bbox=reference_bbox,
        )
        exec_env = {"cq": cq, "cadquery": cq}
        with tracer:
            exec(compiled_code, exec_env)

        states, actions = tracer.get_trajectory_data()
        assert len(actions) > 0, f"Expected at least one action for part {uuid_str}"
        assert len(states) == len(actions) + 1, f"Invariant violation: len(states) != len(actions) + 1"

        traj = {
            "part_id": uuid_str,
            "source_split": "val",
            "source_shard": RAW_VAL_PARQUET.name,
            "source_row": r_idx,
            "num_steps": len(actions),
            "states": states,
            "actions": actions,
        }

        # Validate Schema 0.3.0 invariants
        validate_trajectory(traj)

        # Check topological and image dimensions on initial vs final state
        s0 = states[0]
        sk = states[-1]
        assert s0["images"].shape == (4, 3, 224, 224)
        assert sk["images"].shape == (4, 3, 224, 224)
        assert sk["faces_features"].shape[1] == 32
        assert sk["edges_features"].shape[1] == 16
        assert sk["faces_uv"].shape[1:] == (7, 16, 16)
        assert sk["edges_u"].shape[1:] == (6, 16)

        trajectories.append(traj)

    # 2. Serialize to a fresh test shard
    val_out_dir = tmp_path / "val"
    val_out_dir.mkdir(parents=True, exist_ok=True)
    test_shard_path = val_out_dir / "test_shard_000000.pt"

    shard_dict = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "face_feature_names": list(FACE_FEATURE_NAMES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "command_vocab": CMD2ID,
            "reference_vocab": REF_KIND_TO_NAME,
            "faces_uv_shape": [7, 16, 16],
            "edges_u_shape": [6, 16],
            "images_shape": [4, 3, 224, 224],
            "images_views": ["iso", "front", "top", "right"],
            "images_dtype": "uint8",
            "source_shard": RAW_VAL_PARQUET.name,
            "num_records": len(trajectories),
            "created_at": datetime.datetime.utcnow().isoformat(),
        },
        "records": trajectories,
    }

    validate_shard(shard_dict)
    torch.save(shard_dict, test_shard_path)
    assert test_shard_path.exists() and test_shard_path.stat().st_size > 0

    # 3. Test CADTransitionDataset & DataLoader on freshly generated shard
    dataset = CADTransitionDataset(
        root_dir=tmp_path,
        split="val",
        step_horizon=1,
        normalize_images=True,
    )
    assert len(dataset) > 0, "Dataset should have extracted transition pairs"

    loader = DataLoader(
        dataset,
        batch_size=min(4, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_transition_batch,
    )

    batch = next(iter(loader))
    assert "current_state" in batch
    assert "action" in batch
    assert "next_state" in batch

    # Check finite tensors
    assert torch.all(torch.isfinite(batch["current_state"]["images"]))
    assert torch.all(torch.isfinite(batch["next_state"]["images"]))
    assert torch.all(torch.isfinite(batch["action"]["params"]))
    assert torch.all(torch.isfinite(batch["action"]["profile_points_2d"]))

    # Check batch shapes
    b_size = len(batch["action"]["cmd_id"])
    assert batch["current_state"]["images"].shape == (b_size, 4, 3, 224, 224)
    assert batch["action"]["params"].shape == (b_size, 12)
    assert batch["action"]["param_mask"].shape == (b_size, 12)
    assert batch["action"]["profile_points_2d"].shape == (b_size, 64, 2)
    assert torch.all((batch["action"]["cmd_id"] >= 0) & (batch["action"]["cmd_id"] < NUM_COMMANDS))
