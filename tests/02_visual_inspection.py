"""Visual inspection and CAD artifact export suite for representative samples.

Exports for each sample:
1. original/:
   - code.py: original CadQuery python code
   - final_original.step: raw final STEP file from parquet
   - final_original.stl: raw final STL mesh from parquet
   - metadata.json: parquet row metadata (UUID, ops count, latencies)
2. intermediate_steps/:
   - step_00_empty/ through step_K_<op>/:
     - solid.step: OpenCASCADE in-memory solid exported as true STEP CAD file
     - solid.stl: 3D mesh exported at this exact construction step
     - iso.png, front.png, top.png, right.png: 4-view canonical high-res renders
     - action_info.json: action context, parameters, reference targets
     - state_summary.json: topological face/edge counts, volume, bounding box
3. visualizations/:
   - evolution_contact_sheet.png: full 4-view synchronized transition grid (S_0 -> S_K)
   - construction_history_iso.gif: animated 3D isometric construction timelapse
4. README.md: comprehensive breakdown of the part and step-by-step history
5. trajectory.pt: standalone PyTorch trajectory tensor file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
from src.cad.tracer import CadQueryRuntimeTracer
from src.cad.renderer import HeadlessCadRenderer
from src.data.schema import CANONICAL_VIEW_NAMES, validate_trajectory
from src.data.vocabulary import ID2CMD, REF_KIND_TO_NAME

# Default high-quality representative samples
DEFAULT_TARGET_UUIDS = [
    "76d7fa7c-6de3-4cbe-733e-ea7afbd1db74",  # 5 steps: extrude -> cutThruAll -> hole -> chamfer -> extrude
    "52a13600-375a-1cec-c855-1f1ad580a6c5",  # 7 steps: box -> extrude -> cutBlind -> cskHole -> cskHole -> fillet -> chamfer
]


def to_serializable(val: Any) -> Any:
    """Recursively convert tensors, arrays, vectors, and objects into JSON-serializable primitives."""
    if isinstance(val, torch.Tensor):
        if val.ndim == 0:
            return val.item()
        return val.tolist()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (np.floating, np.integer)):
        return val.item()
    elif isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    elif hasattr(val, "toTuple"):
        return val.toTuple()
    elif isinstance(val, (int, float, str, bool)) or val is None:
        return val
    else:
        return str(val)


def render_contact_sheet(
    states: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    output_path: Path,
    part_id: str,
) -> None:
    """Generate a clean 4-view x K-steps contact sheet showing the geometric progression."""
    num_steps = len(states)
    num_views = len(CANONICAL_VIEW_NAMES)

    fig, axes = plt.subplots(
        nrows=num_views,
        ncols=num_steps,
        figsize=(3.5 * num_steps, 3.5 * num_views),
        dpi=150,
        squeeze=False,
    )

    for step_idx in range(num_steps):
        imgs = states[step_idx]["images"]  # [4, 3, 224, 224] uint8
        if step_idx == 0:
            step_title = "Step 0: S_0 (Empty)"
        else:
            op_name = actions[step_idx - 1].get("operation_name", "op")
            cmd_id = actions[step_idx - 1].get("cmd_id", 0)
            canonical = ID2CMD.get(cmd_id, op_name.upper())
            step_title = f"Step {step_idx}: {canonical}\n({op_name})"

        for view_idx, view_name in enumerate(CANONICAL_VIEW_NAMES):
            ax = axes[view_idx, step_idx]
            img_np = imgs[view_idx].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img_np)
            ax.set_xticks([])
            ax.set_yticks([])

            if view_idx == 0:
                ax.set_title(step_title, fontsize=11, fontweight="bold", pad=8)
            if step_idx == 0:
                ax.set_ylabel(
                    view_name.upper(),
                    fontsize=12,
                    fontweight="bold",
                    rotation=0,
                    labelpad=30,
                    va="center",
                )

    fig.suptitle(
        f"CADynamics: Geometric State Evolution (tau = (S_0, A_0, ..., S_K))\nPart UUID: {part_id}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def render_animated_gif(
    states: List[Dict[str, Any]],
    output_path: Path,
    view_idx: int = 0,
    duration_ms: int = 800,
) -> None:
    """Create an animated GIF showing the 3D construction timelapse."""
    pil_images: List[Image.Image] = []
    for s in states:
        img_np = s["images"][view_idx].permute(1, 2, 0).cpu().numpy()
        pil_images.append(Image.fromarray(img_np))

    if pil_images:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Duplicate last frame 3 times to pause at completion
        frames = pil_images + [pil_images[-1]] * 2
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )


def export_sample(
    raw_record: Dict[str, Any],
    sample_dir: Path,
    renderer: HeadlessCadRenderer,
) -> Dict[str, Any]:
    """Export all debug artifacts for a single CAD sample."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    part_id = str(raw_record["uuid"])

    # 1. Export Original Raw Assets
    orig_dir = sample_dir / "original"
    orig_dir.mkdir(parents=True, exist_ok=True)

    # CadQuery Python code
    raw_code = raw_record["cadquery_file"]
    code_str = raw_code.decode("utf-8") if isinstance(raw_code, bytes) else str(raw_code)
    (orig_dir / "code.py").write_text(code_str, encoding="utf-8")

    # Original STEP from parquet
    if raw_record.get("step_file"):
        step_bytes = raw_record["step_file"]
        if isinstance(step_bytes, str):
            step_bytes = step_bytes.encode("utf-8")
        (orig_dir / "final_original.step").write_bytes(step_bytes)

    # Original STL from parquet
    if raw_record.get("stl_file"):
        stl_bytes = raw_record["stl_file"]
        if isinstance(stl_bytes, str):
            stl_bytes = stl_bytes.encode("utf-8")
        (orig_dir / "final_original.stl").write_bytes(stl_bytes)

    # Parquet Row Metadata & Raw Dataset Renders
    raw_renders_dir = orig_dir / "raw_dataset_renders"
    raw_renders_dir.mkdir(parents=True, exist_ok=True)
    meta_dict: Dict[str, Any] = {}
    for k, v in raw_record.items():
        if k in ("cadquery_file", "step_file", "stl_file"):
            continue
        if k.startswith("image_") and isinstance(v, bytes):
            (raw_renders_dir / f"{k}.png").write_bytes(v)
            meta_dict[k] = f"raw_dataset_renders/{k}.png ({len(v)} bytes)"
        elif isinstance(v, bytes):
            meta_dict[k] = f"<binary blob: {len(v)} bytes>"
        else:
            meta_dict[k] = v

    (orig_dir / "metadata.json").write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    # Pre-pass: compute final bounding box for unified camera framing
    compiled_code = compile(code_str, f"<sample_{part_id}>", "exec")
    reference_bbox = None
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

    # 2. Intercept and Replay CadQuery Execution with locked camera framing
    tracer = CadQueryRuntimeTracer(
        render_images=True,
        renderer=renderer,
        retain_solids=True,
        reference_bbox=reference_bbox,
    )
    exec_env = {"cq": cq, "cadquery": cq}
    with tracer:
        exec(compiled_code, exec_env)

    states, actions = tracer.get_trajectory_data()
    solids = tracer.get_solids()

    assert len(states) == len(actions) + 1, f"Mismatch: {len(states)} states != {len(actions)} actions + 1"
    assert len(solids) == len(actions), f"Mismatch: {len(solids)} solids != {len(actions)} actions"

    # Validate trajectory
    trajectory = {
        "part_id": part_id,
        "source_split": "val",
        "source_shard": raw_record.get("shard_name", "unknown.parquet"),
        "source_row": raw_record.get("row_idx", 0),
        "num_steps": len(actions),
        "states": states,
        "actions": actions,
    }
    validate_trajectory(trajectory)

    # Save standalone .pt file
    torch.save(trajectory, sample_dir / "trajectory.pt")

    # 3. Export Intermediate Steps
    steps_dir = sample_dir / "intermediate_steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    # Step 00: Empty Workspace
    s0_dir = steps_dir / "step_00_empty"
    s0_dir.mkdir(parents=True, exist_ok=True)
    for v_idx, v_name in enumerate(CANONICAL_VIEW_NAMES):
        img_arr = states[0]["images"][v_idx].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img_arr).save(s0_dir / f"{v_name}.png")
    (s0_dir / "state_summary.json").write_text(
        json.dumps({"step": 0, "is_empty": True, "num_faces": 0, "num_edges": 0, "volume": 0.0}, indent=2)
    )

    history_log: List[Dict[str, Any]] = [
        {"step": 0, "operation": "INITIALIZE", "num_faces": 0, "num_edges": 0, "volume": 0.0}
    ]

    # Steps 1 to K: Successive Solis Commits
    for k, (act, state, solid) in enumerate(zip(actions, states[1:], solids), start=1):
        op_name = act.get("operation_name", "op")
        cmd_id = act.get("cmd_id", 0)
        canonical_cmd = ID2CMD.get(cmd_id, op_name.upper())

        step_folder = steps_dir / f"step_{k:02d}_{op_name}"
        step_folder.mkdir(parents=True, exist_ok=True)

        # Export true intermediate STEP & STL CAD files
        step_cad_path = step_folder / "solid.step"
        stl_cad_path = step_folder / "solid.stl"

        try:
            if hasattr(solid, "exportStep"):
                solid.exportStep(str(step_cad_path))
            elif hasattr(solid, "val"):
                solid.val().exportStep(str(step_cad_path))
        except Exception as e:
            (step_folder / "step_export_error.txt").write_text(str(e))

        try:
            if hasattr(solid, "exportStl"):
                solid.exportStl(str(stl_cad_path))
            elif hasattr(solid, "val"):
                solid.val().exportStl(str(stl_cad_path))
        except Exception as e:
            (step_folder / "stl_export_error.txt").write_text(str(e))

        # Export 4-view canonical renders
        for v_idx, v_name in enumerate(CANONICAL_VIEW_NAMES):
            img_arr = state["images"][v_idx].permute(1, 2, 0).cpu().numpy()
            Image.fromarray(img_arr).save(step_folder / f"{v_name}.png")

        # Action context JSON
        action_dict = {
            "step": k,
            "operation_name": op_name,
            "canonical_command": canonical_cmd,
            "cmd_id": cmd_id,
            "ref_kind": REF_KIND_TO_NAME.get(act.get("ref_kind", 0), "UNKNOWN"),
            "params_raw": act.get("params_raw"),
            "params_symlog": act.get("params").tolist() if act.get("params") is not None else None,
            "param_mask": act.get("param_mask").tolist() if act.get("param_mask") is not None else None,
        }
        (step_folder / "action_info.json").write_text(json.dumps(to_serializable(action_dict), indent=2), encoding="utf-8")

        # State summary JSON
        volume = float(solid.Volume()) if hasattr(solid, "Volume") else 0.0
        bb = solid.BoundingBox() if hasattr(solid, "BoundingBox") else None
        bb_dict = {
            "xmin": bb.xmin, "xmax": bb.xmax,
            "ymin": bb.ymin, "ymax": bb.ymax,
            "zmin": bb.zmin, "zmax": bb.zmax,
        } if bb else {}

        state_dict = {
            "step": k,
            "num_faces": int(state["faces_features"].shape[0]),
            "num_edges": int(state["edges_features"].shape[0]),
            "volume": volume,
            "bounding_box": bb_dict,
        }
        (step_folder / "state_summary.json").write_text(json.dumps(to_serializable(state_dict), indent=2), encoding="utf-8")

        history_log.append({
            "step": k,
            "operation": f"{canonical_cmd} ({op_name})",
            "num_faces": state_dict["num_faces"],
            "num_edges": state_dict["num_edges"],
            "volume": round(volume, 3),
        })

    # 4. Generate Visualizations (Contact Sheet & Animated GIF)
    vis_dir = sample_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    contact_sheet_path = vis_dir / "evolution_contact_sheet.png"
    render_contact_sheet(states, actions, contact_sheet_path, part_id=part_id)

    gif_path = vis_dir / "construction_history_iso.gif"
    render_animated_gif(states, gif_path, view_idx=0, duration_ms=800)

    # 5. Generate Markdown Documentation
    readme_content = f"""# Debug Artifacts: Sample `{part_id}`

**UUID**: `{part_id}`  
**Source Parquet**: `{raw_record.get('shard_name', 'unknown')}` (Row {raw_record.get('row_idx', 0)})  
**Total Steps ($K$)**: `{len(actions)}`  
**Generated On**: `{os.getenv('HOSTNAME', 'cadyn-env')}`  

---

## 📸 Geometric Evolution (4-View Canonical Projections)

![Evolution Contact Sheet](visualizations/evolution_contact_sheet.png)

### 3D Isometric Timelapse
![Construction History](visualizations/construction_history_iso.gif)

---

## 🛠 Construction History Table

| Step | Operation | Faces | Edges | Volume (mm³) | Step CAD File |
| :---: | :--- | :---: | :---: | :---: | :--- |
"""
    for item in history_log:
        s_idx = item["step"]
        if s_idx == 0:
            step_link = "`step_00_empty/`"
        else:
            op_fld = f"step_{s_idx:02d}_{actions[s_idx - 1]['operation_name']}"
            step_link = f"[`solid.step`](intermediate_steps/{op_fld}/solid.step)"

        readme_content += (
            f"| {item['step']} | {item['operation']} | {item['num_faces']} | "
            f"{item['num_edges']} | {item['volume']} | {step_link} |\n"
        )

    readme_content += f"""
---

## 📁 Directory Structure
```text
{sample_dir.name}/
├── original/
│   ├── code.py                      # Original CadQuery script
│   ├── final_original.step          # Original final STEP from parquet
│   ├── final_original.stl           # Original final STL mesh from parquet
│   └── metadata.json                # Parquet record metadata
├── intermediate_steps/
│   ├── step_00_empty/               # Neutral gray canvas [4, 3, 224, 224]
│   └── step_01_.../                 # true solid.step, solid.stl, action_info.json, renders
├── visualizations/
│   ├── evolution_contact_sheet.png  # Full 4-view resolution transition strip
│   └── construction_history_iso.gif # 3D isometric animated GIF
├── trajectory.pt                    # Extracted PyTorch trajectory dictionary
└── README.md                        # This inspection report
```
"""
    (sample_dir / "README.md").write_text(readme_content, encoding="utf-8")

    return {
        "uuid": part_id,
        "sample_dir": str(sample_dir),
        "num_steps": len(actions),
        "history": history_log,
    }


def export_debug_samples(
    parquet_path: str = "data/zero_to_cad_100k/data/val/85_5c8bc6f0459e4abba11cfaec2884d1f8_000000_000000-0.parquet",
    output_base_dir: str = "data/debug",
    target_uuids: Optional[List[str]] = None,
    num_samples: int = 2,
) -> List[Dict[str, Any]]:
    """Main function to export representative debug samples."""
    parquet_file = Path(parquet_path)
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading parquet file: {parquet_file}...")
    tbl = pq.read_table(str(parquet_file))
    pylist = tbl.to_pylist()

    # Match target UUIDs or pick first N
    selected_records: List[Dict[str, Any]] = []
    if target_uuids:
        uuid_set = set(target_uuids)
        for row_idx, r in enumerate(pylist):
            if str(r["uuid"]) in uuid_set:
                r["shard_name"] = parquet_file.name
                r["row_idx"] = row_idx
                selected_records.append(r)
                if len(selected_records) == len(target_uuids):
                    break
    else:
        for row_idx, r in enumerate(pylist[:num_samples]):
            r["shard_name"] = parquet_file.name
            r["row_idx"] = row_idx
            selected_records.append(r)

    print(f"Found {len(selected_records)} sample(s) to export.")

    renderer = HeadlessCadRenderer()
    results = []

    for idx, r in enumerate(selected_records, start=1):
        uuid_short = str(r["uuid"])[:8]
        sample_folder = output_dir / f"sample_{idx:02d}_{uuid_short}"
        print(f"\n[{idx}/{len(selected_records)}] Exporting debug sample -> {sample_folder.name}...")
        res = export_sample(r, sample_folder, renderer=renderer)
        results.append(res)
        print(f"  Exported {res['num_steps']} intermediate steps for {res['uuid']} successfully!")

    print(f"\nAll {len(results)} samples exported successfully to {output_dir.resolve()}!")
    return results


def test_visual_inspection() -> None:
    """Pytest-compatible test verifying visual inspection and sample export pipeline."""
    results = export_debug_samples(
        num_samples=2,
        target_uuids=DEFAULT_TARGET_UUIDS,
    )
    assert len(results) == 2, f"Expected 2 exported samples, got {len(results)}"
    for r in results:
        sample_path = Path(r["sample_dir"])
        assert (sample_path / "original/code.py").exists()
        assert (sample_path / "original/final_original.step").exists()
        assert (sample_path / "visualizations/evolution_contact_sheet.png").exists()
        assert (sample_path / "visualizations/construction_history_iso.gif").exists()
        assert (sample_path / "trajectory.pt").exists()
        assert (sample_path / "README.md").exists()
        # Verify intermediate steps have solid.step
        for step_dir in (sample_path / "intermediate_steps").glob("step_0[1-9]*"):
            assert (step_dir / "solid.step").exists()
            assert (step_dir / "iso.png").exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export complete debug samples with intermediate CAD and renders.")
    parser.add_argument("--parquet", type=str, default="data/zero_to_cad_100k/data/val/85_5c8bc6f0459e4abba11cfaec2884d1f8_000000_000000-0.parquet")
    parser.add_argument("--output-dir", type=str, default="data/debug")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--uuids", nargs="*", default=DEFAULT_TARGET_UUIDS)
    args = parser.parse_args()

    export_debug_samples(
        parquet_path=args.parquet,
        output_base_dir=args.output_dir,
        target_uuids=args.uuids if args.uuids else None,
        num_samples=args.num_samples,
    )
