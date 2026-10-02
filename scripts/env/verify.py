#!/usr/bin/env python3
"""scripts/env/verify.py
Verifies the CADynamics environment integrity.

Checks:
1. Core CAD and geometry stack: CadQuery, OCP, OpenCASCADE kernel operation.
2. Research & ML stack: PyTorch, NumPy, Pandas, PyArrow.
3. Functional CAD kernel test: Constructs a 3D parametric solid (box) and checks volume/topology.
4. GPU / Accelerator availability: Reported clearly (non-fatal unless --require-gpu is passed).

Exit code:
- 0: All critical requirements satisfied
- 1: One or more critical requirements failed
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple


def check_module(name: str) -> Tuple[bool, str]:
    """Attempt importing module and retrieve its version or error."""
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "version unavailable")
        return True, str(ver)
    except Exception as exc:
        return False, str(exc)


def test_cad_kernel() -> Tuple[bool, str]:
    """Test actual B-Rep solid construction using OpenCASCADE via CadQuery."""
    try:
        import cadquery as cq
        box = cq.Workplane("XY").box(10.0, 20.0, 30.0)
        solid = box.val()
        vol = solid.Volume()
        num_faces = len(box.faces().vals())
        expected_vol = 6000.0
        if abs(vol - expected_vol) > 1e-4:
            return False, f"Volume mismatch: {vol:.2f} mm^3 (expected {expected_vol:.2f})"
        if num_faces != 6:
            return False, f"Face count mismatch: {num_faces} (expected 6)"
        return True, f"Box volume={vol:.1f} mm^3, faces={num_faces}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CADynamics environment integrity.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA GPU is not accessible")
    args = parser.parse_args()

    print("=" * 68)
    print("CADynamics Environment Verification")
    print("=" * 68)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print("-" * 68)

    critical_failures: List[str] = []

    # 1. Critical Module Imports
    required_packages = [
        "cadquery",
        "OCP",
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
    ]

    print("Checking Critical Modules:")
    for pkg in required_packages:
        ok, detail = check_module(pkg)
        status = "[ OK ]" if ok else "[FAIL]"
        print(f"  {status} {pkg:<12} : {detail}")
        if not ok:
            critical_failures.append(f"Module import failed: {pkg} ({detail})")

    # Optional CAD/viz modules
    optional_packages = ["OCC", "pyvista", "vtk", "pydantic", "pytest"]
    print("\nChecking Auxiliary CAD / Viz / Dev Modules:")
    for pkg in optional_packages:
        ok, detail = check_module(pkg)
        status = "[ OK ]" if ok else "[INFO]"
        print(f"  {status} {pkg:<12} : {detail}")

    # 2. Functional CAD Kernel Test
    print("\nValidating CAD Kernel (B-Rep Solid Generation):")
    cad_ok, cad_msg = test_cad_kernel()
    if cad_ok:
        print(f"  [ OK ] OpenCASCADE kernel : {cad_msg}")
    else:
        print(f"  [FAIL] OpenCASCADE kernel : {cad_msg}")
        critical_failures.append(f"CAD kernel geometry test failed: {cad_msg}")

    # 3. Accelerator / GPU Check
    print("\nChecking Compute Accelerator (CUDA):")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda or "none"
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            print(f"  [ OK ] CUDA Available     : True (Version {cuda_ver}, Device: {device_name})")
        else:
            msg = f"CUDA Available     : False (PyTorch built with CUDA {cuda_ver})"
            if args.require_gpu:
                print(f"  [FAIL] {msg} - GPU strictly required!")
                critical_failures.append("GPU not available but --require-gpu was specified")
            else:
                print(f"  [INFO] {msg} (CPU execution mode active)")
    except Exception as exc:
        print(f"  [WARN] CUDA check error: {exc}")

    print("=" * 68)
    if critical_failures:
        print("ENVIRONMENT VERIFICATION FAILED!")
        for fail in critical_failures:
            print(f"  - {fail}")
        print("=" * 68)
        return 1

    print("ALL CRITICAL ENVIRONMENT CHECKS PASSED SUCCESSFULLY.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
