"""Production offline preprocessor extracting true intermediate CAD trajectories.

Constructs trustworthy dataset of causal transition trajectories:
    tau = (S_0, A_0, S_1, A_1, ..., A_{K-1}, S_K)

Features:
  - Enforces strict S_{k+1} = F(S_k, A_k) invariant
  - In-memory deterministic rendering (no intermediate files written)
  - Isolated worker processes using multiprocessing spawn context
  - 15-second per-part timeout protection
  - Comprehensive failure accounting and structured logging
  - Chunked .pt shard serialization with versioned metadata
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

# Ensure repository root is on sys.path
# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Suppress VTK hardware capability warnings and stderr spam before OpenGL initialization
import vtk

if hasattr(vtk, "vtkLogger"):
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
vtk.vtkObject.GlobalWarningDisplayOff()

# Initialize off-screen PyVista OpenGL context before importing CadQuery/OCP
import pyvista as pv

try:
    _early_plotter = pv.Plotter(off_screen=True, window_size=[10, 10])
    _early_plotter.render()
    _early_plotter.screenshot(transparent_background=False)
    _early_plotter.close()
except Exception:
    pass

import cadquery as cq
import OCP

from src.cad.tracer import CadQueryRuntimeTracer
from src.cad.renderer import HeadlessCadRenderer
from src.data.schema import (
    EDGE_FEATURE_NAMES,
    FACE_FEATURE_NAMES,
    SCHEMA_VERSION,
    validate_trajectory,
)
from src.data.vocabulary import (
    CMD2ID,
    COMMAND_VOCAB,
    NUM_COMMANDS,
    REF_KIND_TO_NAME,
    VOCABULARY_VERSION,
)


def _worker_process_part(
    code_str: str,
    part_meta: Dict[str, Any],
    render_images: bool = True,
    uv_grid_size: int = 16,
    curve_samples: int = 16,
    image_size: int = 224,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Worker task executing inside an isolated spawned process.

    Returns:
        status: One of ('success', 'syntax_error', 'runtime_error', 'unsupported_operation',
                        'invalid_geometry', 'extraction_error', 'render_error')
        trajectory: Extracted and validated trajectory dictionary (if success)
        error_msg: String description of failure (if failed)
    """
    tracer = None
    try:
        # 1. Syntax check
        try:
            compiled_code = compile(code_str, "<cadquery_program>", "exec")
        except SyntaxError as se:
            return "syntax_error", None, f"SyntaxError: {se}"

        # 2. Runtime execution inside CadQueryRuntimeTracer
        # Pre-pass: evaluate geometry to compute final bounding box for unified camera framing
        reference_bbox = None
        if render_images:
            try:
                pre_env = {"cq": cq, "cadquery": cq}
                exec(compiled_code, pre_env)
                target = pre_env.get("result") or pre_env.get("part")
                if target is None:
                    for val in pre_env.values():
                        if isinstance(val, cq.Workplane) and val.val() is not None:
                            target = val
                            break
                if isinstance(target, cq.Workplane) and target.val() is not None:
                    bb = target.val().BoundingBox()
                    reference_bbox = [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax]
            except Exception:
                reference_bbox = None

        renderer = HeadlessCadRenderer(image_size=image_size) if render_images else None
        tracer = CadQueryRuntimeTracer(
            render_images=render_images,
            renderer=renderer,
            reference_bbox=reference_bbox,
            uv_grid_size=uv_grid_size,
            curve_samples=curve_samples,
        )

        exec_env = {
            "cq": cq,
            "cadquery": cq,
        }

        with tracer:
            exec(compiled_code, exec_env)

        states, actions = tracer.get_trajectory_data()

        # Invariant: len(states) == len(actions) + 1
        num_steps = len(actions)
        if num_steps == 0:
            return "invalid_geometry", None, "No solid-commit actions intercepted"

        trajectory = {
            "part_id": part_meta["uuid"],
            "source_split": part_meta.get("split", "train"),
            "source_shard": part_meta.get("shard_name", "unknown"),
            "source_row": part_meta.get("row_idx", 0),
            "num_steps": num_steps,
            "states": states,
            "actions": actions,
        }

        # 3. Invariant validation
        try:
            validate_trajectory(trajectory)
        except Exception as ve:
            return "extraction_error", None, f"Validation failure: {ve}"

        return "success", trajectory, None

    except Exception as e:
        err_str = str(e)
        # Classify unsupported operations
        if "unsupported" in err_str.lower():
            return "unsupported_operation", None, err_str
        return "runtime_error", None, f"{type(e).__name__}: {err_str}"


def _worker_entrypoint(
    task_args: Tuple[str, Dict[str, Any], bool, float, int, int, int]
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], float]:
    """Top-level worker function with execution timing."""
    code_str, part_meta, render_images, _, uv_grid_size, curve_samples, image_size = task_args
    t_start = time.time()
    try:
        status, trajectory, err = _worker_process_part(
            code_str,
            part_meta,
            render_images,
            uv_grid_size=uv_grid_size,
            curve_samples=curve_samples,
            image_size=image_size,
        )
    except Exception as e:
        status, trajectory, err = "runtime_error", None, str(e)
    t_dur = time.time() - t_start
    return status, trajectory, err, t_dur


def save_shard(
    records: List[Dict[str, Any]],
    output_path: Path,
    metadata_extra: Optional[Dict[str, Any]] = None,
    uv_grid_size: int = 16,
    curve_samples: int = 16,
    image_size: int = 224,
) -> int:
    """Save chunk of trajectories into a versioned .pt shard file matching parquet stem."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_meta = {
        "face_feature_names": list(FACE_FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "command_vocab": CMD2ID,
        "reference_vocab": REF_KIND_TO_NAME,
        "faces_uv_shape": [7, uv_grid_size, uv_grid_size],
        "edges_u_shape": [6, curve_samples],
        "images_shape": [4, 3, image_size, image_size],
        "images_views": ["iso", "front", "top", "right"],
        "images_dtype": "uint8",
        "source_shard": output_path.stem + ".parquet",
        "num_records": len(records),
        "cadquery_version": getattr(cq, "__version__", "unknown"),
        "ocp_version": getattr(OCP, "__version__", "7.8.1.2"),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    if metadata_extra:
        shard_meta.update(metadata_extra)

    shard_data = {
        "schema_version": SCHEMA_VERSION,
        "metadata": shard_meta,
        "records": records,
    }
    torch.save(shard_data, output_path)
    return os.path.getsize(output_path)


def deterministic_split(part_id: str, train_ratio: float = 0.9) -> str:
    """Deterministically route a part_id into 'train' (~90%) or 'val' (~10%).

    Uses MD5 hash of part_id to guarantee repeatable, leakage-free assignment
    when operating on unified pools without pre-existing splits.
    """
    h = int(hashlib.md5(part_id.encode("utf-8")).hexdigest()[:8], 16)
    return "train" if (h % 100) < int(train_ratio * 100) else "val"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Zero-to-CAD into true intermediate state trajectories."
    )
    parser.add_argument("--max-parts", type=int, default=1000, help="Maximum parts to process (-1 for all).")
    parser.add_argument("--max-shards", type=int, default=None, help="Maximum number of parquet shards to process.")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker processes.")
    parser.add_argument("--input-dir", type=str, default="data/zero_to_cad_100k", help="Input root.")
    parser.add_argument("--output-dir", type=str, default="data/processed_data", help="Output root directory.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log output directory.")
    parser.add_argument("--split", type=str, default="val", help="Data split to process ('train', 'val', 'test', or 'all').")
    parser.add_argument("--shard-size", type=int, default=None, help="Deprecated (shards map 1-to-1 to parquet files).")
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout per part in seconds.")
    parser.add_argument("--no-render", action="store_true", help="Skip rendering for fast dry runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt shards.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio for unified pool hashing.")
    parser.add_argument("--uv-grid-size", type=int, default=16, help="Discretization grid size for face UV parameter space.")
    parser.add_argument("--curve-samples", type=int, default=16, help="Discretization samples along 3D edge curves.")
    parser.add_argument("--image-size", type=int, default=224, help="Rendered multi-view image resolution (H=W).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"preprocess_{timestamp}.log"

    # Setup file and stdout logging
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 75)
    logger.info("CADYNAMICS: Intermediate State & Trajectory Preprocessing")
    logger.info("=" * 75)
    logger.info(f"Schema Version:     {SCHEMA_VERSION}")
    logger.info(f"Vocabulary Version: {VOCABULARY_VERSION} ({NUM_COMMANDS} commands)")
    logger.info(f"Input Directory:    {input_path.resolve()}")
    logger.info(f"Output Directory:   {output_path.resolve()}")
    logger.info(f"Target Split:       {args.split}")
    logger.info(f"Max Parts:          {args.max_parts if args.max_parts > 0 else 'unlimited'}")
    logger.info(f"Max Shards:         {args.max_shards if args.max_shards is not None else 'unlimited'}")
    logger.info(f"Num Workers:        {args.num_workers}")
    logger.info(f"Per-Part Timeout:   {args.timeout}s")
    logger.info(f"Render Images:      {not args.no_render}")
    logger.info(f"UV Grid Size:       {args.uv_grid_size}x{args.uv_grid_size}")
    logger.info(f"Curve Samples:      {args.curve_samples}")
    logger.info(f"Image Resolution:   {args.image_size}x{args.image_size}")
    logger.info(f"Overwrite:          {args.overwrite}")
    logger.info("=" * 75)

    # 1. Discover Parquet shards
    explicit_split_dirs: Dict[str, List[Path]] = {}
    for s_name in ("train", "val", "test"):
        d = input_path / "data" / s_name
        if not d.exists():
            d = input_path / s_name
        if d.exists() and d.is_dir():
            found_files = sorted(list(d.glob("*.parquet")))
            if found_files:
                explicit_split_dirs[s_name] = found_files

    shards_to_process: List[Tuple[str, Path]] = []
    if explicit_split_dirs:
        splits_to_load = [args.split] if args.split in explicit_split_dirs else (
            list(explicit_split_dirs.keys()) if args.split == "all" else []
        )
        if not splits_to_load:
            logger.error(
                f"Requested split '{args.split}' not found among explicit splits: {list(explicit_split_dirs.keys())}"
            )
            sys.exit(1)

        for s_name in splits_to_load:
            p_files = explicit_split_dirs[s_name]
            logger.info(f"Found {len(p_files)} parquet shards for explicit split '{s_name}'.")
            for pf in p_files:
                shards_to_process.append((s_name, pf))
    else:
        all_parquet_files = sorted(list(input_path.glob("**/*.parquet")))
        if not all_parquet_files:
            logger.error(f"No parquet shards found under {input_path}")
            sys.exit(1)
        logger.info(f"Unified parquet pool: found {len(all_parquet_files)} parquet files.")
        for pf in all_parquet_files:
            shards_to_process.append((args.split if args.split != "all" else "train", pf))

    if args.max_shards is not None and args.max_shards > 0:
        shards_to_process = shards_to_process[:args.max_shards]

    logger.info(f"Queued {len(shards_to_process)} parquet shards for 1-to-1 processing.")

    # 2. Setup counters & worker pool
    failure_counts = {
        "syntax_error": 0,
        "timeout": 0,
        "runtime_error": 0,
        "unsupported_operation": 0,
        "invalid_geometry": 0,
        "extraction_error": 0,
        "render_error": 0,
    }
    command_counts: Dict[str, int] = {cmd: 0 for cmd in COMMAND_VOCAB}
    unsupported_counts: Dict[str, int] = {}
    durations: List[float] = []

    saved_shards_by_split: Dict[str, int] = collections.defaultdict(int)
    total_parts_attempted = 0
    total_parts_succeeded = 0
    total_output_bytes = 0

    t_global_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.num_workers) as pool:
        for shard_idx, (s_name, pf_path) in enumerate(shards_to_process):
            if args.max_parts > 0 and total_parts_attempted >= args.max_parts:
                logger.info(f"Reached --max-parts limit ({args.max_parts}). Stopping.")
                break

            split_dir = output_path / s_name
            split_dir.mkdir(parents=True, exist_ok=True)
            target_pt = split_dir / f"{pf_path.stem}.pt"

            if target_pt.exists() and not args.overwrite:
                if target_pt.stat().st_size > 0:
                    logger.info(
                        f"[{shard_idx + 1}/{len(shards_to_process)}] Skipping {target_pt.name} "
                        f"(already exists, {target_pt.stat().st_size / (1024*1024):.1f} MB)"
                    )
                    continue

            # Read Parquet
            tbl = pq.read_table(str(pf_path), columns=["uuid", "cadquery_file", "cadquery_ops_count"])
            pylist = tbl.to_pylist()

            # Cap parts if max_parts is set
            if args.max_parts > 0:
                remaining_parts = args.max_parts - total_parts_attempted
                pylist = pylist[:remaining_parts]

            shard_tasks = []
            for row_idx, r in enumerate(pylist):
                raw_code = r["cadquery_file"]
                code_str = raw_code.decode("utf-8") if isinstance(raw_code, bytes) else str(raw_code)
                meta = {
                    "uuid": str(r["uuid"]),
                    "split": s_name,
                    "shard_name": pf_path.name,
                    "row_idx": row_idx,
                    "ops_count": r.get("cadquery_ops_count", 0),
                }
                shard_tasks.append(
                    (
                        code_str,
                        meta,
                        not args.no_render,
                        args.timeout,
                        args.uv_grid_size,
                        args.curve_samples,
                        args.image_size,
                    )
                )

            if not shard_tasks:
                continue

            logger.info(
                f"[{shard_idx + 1}/{len(shards_to_process)}] Processing shard '{pf_path.name}' "
                f"({len(shard_tasks)} parts) -> '{target_pt.name}'..."
            )

            # Submit tasks for this shard
            async_results = [
                (task[1]["uuid"], pool.apply_async(_worker_entrypoint, (task,)))
                for task in shard_tasks
            ]

            shard_trajectories = []
            shard_successes = 0

            for i, (part_id, res) in enumerate(async_results):
                total_parts_attempted += 1
                try:
                    status, trajectory, err_msg, dur = res.get(timeout=args.timeout)
                except mp.TimeoutError:
                    status = "timeout"
                    trajectory = None
                    err_msg = f"Timed out after {args.timeout}s"
                    dur = args.timeout
                except Exception as e:
                    status = "runtime_error"
                    trajectory = None
                    err_msg = str(e)
                    dur = args.timeout

                durations.append(dur)

                if status == "success" and trajectory is not None:
                    shard_trajectories.append(trajectory)
                    total_parts_succeeded += 1
                    shard_successes += 1

                    for a in trajectory["actions"]:
                        c_id = a.get("cmd_id", 0)
                        if 0 <= c_id < NUM_COMMANDS:
                            command_counts[COMMAND_VOCAB[c_id]] += 1
                else:
                    failure_counts[status] = failure_counts.get(status, 0) + 1
                    if status == "unsupported_operation" and err_msg:
                        unsupported_counts[err_msg] = unsupported_counts.get(err_msg, 0) + 1

                if (i + 1) % max(1, len(shard_tasks) // 5) == 0 or (i + 1) == len(shard_tasks):
                    cur_rate = (shard_successes / (i + 1)) * 100.0
                    logger.info(
                        f"  Shard Progress: {i + 1}/{len(shard_tasks)} parts ({((i + 1) / len(shard_tasks))*100:.1f}%) | "
                        f"Successes: {shard_successes} ({cur_rate:.1f}%) | "
                        f"Timeouts: {failure_counts['timeout']} | "
                        f"Avg: {np.mean(durations[-len(shard_tasks):]):.2f}s/part"
                    )

            # Save this shard if any trajectories succeeded
            if shard_trajectories:
                sz = save_shard(
                    shard_trajectories,
                    target_pt,
                    metadata_extra={
                        "source_shard": pf_path.name,
                        "split": s_name,
                        "total_parts_in_parquet": len(tbl),
                        "attempted_parts": len(shard_tasks),
                        "num_successful_trajectories": len(shard_trajectories),
                    },
                    uv_grid_size=args.uv_grid_size,
                    curve_samples=args.curve_samples,
                    image_size=args.image_size,
                )
                total_output_bytes += sz
                saved_shards_by_split[s_name] += 1
                logger.info(
                    f"Saved [{s_name}] shard {target_pt.name} with "
                    f"{len(shard_trajectories)}/{len(shard_tasks)} trajectories "
                    f"({sz / (1024*1024):.2f} MB)"
                )
            else:
                logger.warning(
                    f"No successful trajectories for shard '{pf_path.name}'. Target file not created."
                )

    t_total_elapsed = time.time() - t_global_start

    # 3. Compute Summary Statistics
    total_attempted = total_parts_attempted
    total_success = total_parts_succeeded
    success_rate = (total_success / total_attempted * 100.0) if total_attempted > 0 else 0.0

    total_transitions = sum(command_counts.values())
    mean_transitions = (total_transitions / total_success) if total_success > 0 else 0.0

    avg_time = float(np.mean(durations)) if durations else 0.0
    med_time = float(np.median(durations)) if durations else 0.0
    p95_time = float(np.percentile(durations, 95)) if durations else 0.0

    # 5. Output Structured Report to Log
    logger.info("=" * 75)
    logger.info("PREPROCESSING SUMMARY REPORT")
    logger.info("=" * 75)
    logger.info(f"Total parts attempted:         {total_attempted}")
    logger.info(f"Successful trajectories:       {total_success}")
    logger.info(f"Success rate:                  {success_rate:.2f}%")
    logger.info("-" * 75)
    logger.info("Failure Breakdown:")
    for f_cat, f_cnt in failure_counts.items():
        pct = (f_cnt / total_attempted * 100.0) if total_attempted > 0 else 0.0
        logger.info(f"  {f_cat:25s}: {f_cnt:5d} ({pct:5.2f}%)")
    logger.info("-" * 75)
    logger.info("Split Shard Breakdown:")
    for s_name, s_count in saved_shards_by_split.items():
        logger.info(f"  {s_name:10s}: {s_count} shards -> {output_path / s_name}")
    logger.info("-" * 75)
    logger.info(f"Total extracted transitions:   {total_transitions}")
    logger.info(f"Mean transitions / trajectory: {mean_transitions:.2f}")
    logger.info(f"Average extraction time:       {avg_time:.3f} s/part")
    logger.info(f"Median extraction time:        {med_time:.3f} s/part")
    logger.info(f"p95 extraction time:           {p95_time:.3f} s/part")
    logger.info(f"Total processing wall time:    {t_total_elapsed:.1f} s")
    logger.info("-" * 75)
    logger.info("Command Token Distribution:")
    for cmd, cnt in sorted(command_counts.items(), key=lambda x: x[1], reverse=True):
        if cnt > 0:
            pct = (cnt / total_transitions * 100.0) if total_transitions > 0 else 0.0
            logger.info(f"  {cmd:20s}: {cnt:5d} ({pct:5.2f}%)")
    logger.info("-" * 75)
    logger.info(f"Total output shards:           {sum(saved_shards_by_split.values())}")
    logger.info(f"Total output size:             {total_output_bytes / (1024*1024):.2f} MB")
    logger.info(f"Log written to:                {log_file.resolve()}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
