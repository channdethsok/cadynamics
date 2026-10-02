"""Verification and benchmark test for CADynamics DataLoader.

Verifies:
  - Multi-worker and single-worker execution
  - Throughput measurement (samples/sec)
  - Schema 0.3.0 output shapes:
      * action['params'].shape == (B, 12)
      * action['param_mask'].shape == (B, 12)
      * action['cmd_id'].shape == (B,)
      * action['profile_points_2d'].shape == (B, 64, 2)
      * current_state['images'].shape == (B, 4, 3, 224, 224)
      * next_state['images'].shape == (B, 4, 3, 224, 224)
      * faces_features.shape[1] == 32
      * edges_features.shape[1] == 16
      * faces_adjacency_index.shape[0] == 2
  - Command IDs strictly within vocabulary bounds [0, NUM_COMMANDS - 1]
  - All tensors are 100% finite (no NaNs, no Infs)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CADTransitionDataset,
    NUM_COMMANDS,
    collate_transition_batch,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_dataloader")


def verify_batch(batch: dict, expected_batch_size: int) -> None:
    """Perform rigorous checks on a batched dictionary of transition pairs."""
    B = expected_batch_size

    # 1. Action shape checks
    assert "action" in batch, "Missing 'action' key in batch"
    action = batch["action"]
    assert "params" in action, "Missing 'params' in action"
    assert "param_mask" in action, "Missing 'param_mask' in action"
    assert "cmd_id" in action, "Missing 'cmd_id' in action"
    assert "profile_points_2d" in action, "Missing 'profile_points_2d' in action"
    assert "selected_faces_mask" in action, "Missing 'selected_faces_mask' in action"
    assert "selected_edges_mask" in action, "Missing 'selected_edges_mask' in action"

    params = action["params"]
    masks = action["param_mask"]
    cmd_ids = action["cmd_id"]
    profiles = action["profile_points_2d"]
    sel_faces = action["selected_faces_mask"]
    sel_edges = action["selected_edges_mask"]

    assert params.shape == (B, 12), f"Expected params shape {(B, 12)}, got {params.shape}"
    assert masks.shape == (B, 12), f"Expected param_mask shape {(B, 12)}, got {masks.shape}"
    assert cmd_ids.shape == (B,), f"Expected cmd_id shape {(B,)}, got {cmd_ids.shape}"
    assert profiles.shape == (B, 64, 2), f"Expected profile shape {(B, 64, 2)}, got {profiles.shape}"
    assert sel_faces.dtype == torch.bool, f"Expected bool for selected_faces_mask, got {sel_faces.dtype}"
    assert sel_edges.dtype == torch.bool, f"Expected bool for selected_edges_mask, got {sel_edges.dtype}"

    # 2. Command IDs strictly within bounds
    assert torch.all(cmd_ids >= 0) and torch.all(
        cmd_ids < NUM_COMMANDS
    ), f"Command IDs out of vocabulary bounds [0, {NUM_COMMANDS - 1}]: {cmd_ids}"

    # 3. State images shape checks
    curr_state = batch["current_state"]
    next_state = batch["next_state"]

    assert curr_state["images"].shape == (
        B,
        4,
        3,
        224,
        224,
    ), f"Expected current_state images shape {(B, 4, 3, 224, 224)}, got {curr_state['images'].shape}"
    assert next_state["images"].shape == (
        B,
        4,
        3,
        224,
        224,
    ), f"Expected next_state images shape {(B, 4, 3, 224, 224)}, got {next_state['images'].shape}"

    # 4. Graph structure checks
    for state_name, state in [("current_state", curr_state), ("next_state", next_state)]:
        assert "faces_features" in state, f"Missing 'faces_features' in {state_name}"
        assert "edges_features" in state, f"Missing 'edges_features' in {state_name}"
        assert "faces_adjacency_index" in state, f"Missing 'faces_adjacency_index' in {state_name}"
        assert "batch_faces" in state, f"Missing 'batch_faces' in {state_name}"
        assert "batch_edges" in state, f"Missing 'batch_edges' in {state_name}"

        faces = state["faces_features"]
        edges = state["edges_features"]
        adj = state["faces_adjacency_index"]
        batch_f = state["batch_faces"]
        batch_e = state["batch_edges"]

        assert faces.ndim == 2 and faces.shape[1] == 32, f"Expected face feature dim 32, got {faces.shape}"
        assert edges.ndim == 2 and edges.shape[1] == 16, f"Expected edge feature dim 16, got {edges.shape}"
        assert adj.ndim == 2 and adj.shape[0] == 2, f"Expected adjacency dim 2, got {adj.shape}"
        assert batch_f.shape == (faces.shape[0],), f"Mismatch between batch_faces and faces count"
        assert batch_e.shape == (edges.shape[0],), f"Mismatch between batch_edges and edges count"

    # 5. Check finite numbers across all numerical tensors
    def assert_finite(t: torch.Tensor, name: str) -> None:
        if t.is_floating_point():
            assert torch.all(torch.isfinite(t)), f"Tensor {name} contains non-finite values (NaN / Inf)!"

    assert_finite(params, "action.params")
    assert_finite(profiles, "action.profile_points_2d")
    assert_finite(curr_state["images"], "current_state.images")
    assert_finite(next_state["images"], "next_state.images")
    assert_finite(curr_state["faces_features"], "current_state.faces_features")
    assert_finite(next_state["faces_features"], "next_state.faces_features")
    assert_finite(curr_state["edges_features"], "current_state.edges_features")
    assert_finite(next_state["edges_features"], "next_state.edges_features")


def run_benchmark(
    dataset_dir: str = "data/processed_data",
    split: str = "val",
    batch_size: int = 8,
    num_workers: int = 2,
    max_batches: int = 15,
) -> None:
    logger.info("=" * 70)
    logger.info("CADYNAMICS DATALOADER BENCHMARK & VERIFICATION")
    logger.info(f"Root: {dataset_dir} | Split: {split} | Batch size: {batch_size} | Workers: {num_workers}")
    logger.info("=" * 70)

    dataset = CADTransitionDataset(
        root_dir=dataset_dir,
        split=split,
        normalize_images=True,
    )
    logger.info(f"Loaded {len(dataset)} total transition pairs.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_transition_batch,
        drop_last=False,
    )

    total_samples = 0
    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        actual_b = batch["action"]["cmd_id"].shape[0]
        verify_batch(batch, expected_batch_size=actual_b)
        total_samples += actual_b

        if batch_idx == 0:
            first_batch_time = time.perf_counter() - start_time
            logger.info(f"Time to first batch: {first_batch_time:.3f} s")

        if batch_idx + 1 >= max_batches:
            break

    elapsed = time.perf_counter() - start_time
    throughput = total_samples / elapsed if elapsed > 0 else 0.0

    logger.info("-" * 70)
    logger.info(f"Benchmark Results:")
    logger.info(f"  Processed {total_samples} samples across {batch_idx + 1} batches.")
    logger.info(f"  Total time: {elapsed:.3f} s")
    logger.info(f"  Throughput: {throughput:.1f} samples/sec")
    logger.info("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (100% finite, valid schema 0.3.0).")
    logger.info("=" * 70)


def _ensure_fixture_shard(target_dir: Path) -> Path:
    """Ensure a valid test shard exists in target_dir / 'val'."""
    val_dir = target_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    shard_file = val_dir / "fixture_shard.pt"
    if shard_file.exists():
        return target_dir

    from src.data.schema import (
        EDGE_FEATURE_NAMES,
        FACE_FEATURE_NAMES,
        SCHEMA_VERSION,
        create_empty_state,
    )
    from src.data.vocabulary import CMD2ID, REF_KIND_TO_NAME

    records = []
    for uid in ["fixture_part_001", "fixture_part_002"]:
        s0 = create_empty_state()
        s1 = create_empty_state()
        s1["num_faces"] = 1
        s1["num_edges"] = 1
        s1["faces_features"] = torch.zeros((1, 32), dtype=torch.float32)
        s1["faces_uv"] = torch.zeros((1, 7, 16, 16), dtype=torch.float32)
        s1["edges_features"] = torch.zeros((1, 16), dtype=torch.float32)
        s1["edges_u"] = torch.zeros((1, 6, 16), dtype=torch.float32)
        s1["faces_adjacency_index"] = torch.zeros((2, 0), dtype=torch.long)
        s1["faces_adjacency_edge_indices"] = torch.zeros(0, dtype=torch.long)
        s1["edges_features_directed"] = torch.zeros((0, 16), dtype=torch.float32)
        s1["face_edge_index"] = torch.zeros((2, 0), dtype=torch.long)
        s1["reverse_edge_indices"] = torch.zeros(0, dtype=torch.long)
        s1["images"] = torch.full((4, 3, 224, 224), 200, dtype=torch.uint8)

        action = {
            "cmd_id": 6,  # BOX
            "params": torch.zeros(12, dtype=torch.float32),
            "param_mask": torch.ones(12, dtype=torch.float32),
            "ref_kind": 1,
            "ref_entity_indices": torch.zeros(0, dtype=torch.long),
            "ref_points": torch.zeros((0, 3), dtype=torch.float32),
            "ref_directions": torch.zeros((0, 3), dtype=torch.float32),
            "profile_points_2d": torch.zeros((64, 2), dtype=torch.float32),
            "operation_name": "box",
        }
        records.append({
            "part_id": uid,
            "source_split": "val",
            "source_shard": "fixture_shard.pt",
            "source_row": 0,
            "num_steps": 1,
            "states": [s0, s1],
            "actions": [action],
        })

    shard_data = {
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
            "source_shard": "fixture_shard.parquet",
            "num_records": len(records),
        },
        "records": records,
    }
    torch.save(shard_data, shard_file)
    return target_dir


def test_dataloader_batch(tmp_path: Path) -> None:
    """Pytest-compatible test verifying DataLoader batch extraction and schema invariants."""
    # Use production preprocessed shards if available; otherwise use self-contained sandbox fixture
    val_dir = Path("data/processed_data/val")
    if val_dir.exists() and list(val_dir.glob("*.pt")):
        dataset_dir = "data/processed_data"
        batch_size = 4
    else:
        dataset_dir = str(_ensure_fixture_shard(tmp_path))
        batch_size = 2

    run_benchmark(
        dataset_dir=dataset_dir,
        split="val",
        batch_size=batch_size,
        num_workers=0,
        max_batches=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CADynamics DataLoader verification script.")
    parser.add_argument("--root-dir", type=str, default="data/processed_data")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.root_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
