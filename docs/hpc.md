# HPC Transition & Deployment Strategy (Jean Zay & Supercomputing Clusters)

This document describes how CADynamics separates CPU-bound CAD preprocessing from GPU-accelerated large-scale model training, and details the portable software architecture required for supercomputing centers such as Jean Zay (IDRIS / CNRS).

---

## 1. Separation of Concerns: Preprocessing vs. Training

A major architectural challenge in Geometric CAD Machine Learning is that CAD kernels (OpenCASCADE, CadQuery) are complex C++ runtimes that require dynamic shared libraries, specific GL/Mesa contexts for rendering, and Python bindings that may conflict with high-performance MPI or CUDA runtimes on compute nodes.

CADynamics explicitly decouples the pipeline into two independent stages:

```mermaid
flowchart TD
    subgraph Preprocessing ["Stage 1: Preprocessing & Sharding (CPU / Viz Nodes)"]
        raw["Raw Zero-to-CAD Dataset (.parquet / code)"]
        cq["CadQuery Runtime Tracer & OCP Extractor"]
        renderer["HeadlessCadRenderer (PyVista / Mesa / VTK)"]
        shards["Intermediate Trajectory Shards (.parquet / .pt)"]
        raw --> cq
        cq --> renderer
        renderer --> shards
    end

    subgraph Training ["Stage 2: Large-Scale Training (Jean Zay GPU Nodes)"]
        shards --> dataset["ShardedTrajectoryDataset (PyTorch / PyArrow)"]
        dataset --> graph["B-Rep Geometry & Graph Encoder"]
        dataset --> vision["ViT Multi-View Spatial Encoder"]
        vision --> model["Action-JEPA-CAD Dynamics Model"]
    end
```

### Key Architectural Properties
- **Zero Runtime CAD Execution during Training**:
  During model training, the network ingests pre-computed multi-view image tensors (`[4, 3, 224, 224]`), B-Rep Face-Adjacency Graph (FAG) topologies, and action parameter vectors serialized in PyArrow parquet tables or PyTorch `.pt` files.
- **Standalone Training Environment**:
  Compute nodes executing training jobs do NOT need CadQuery, OpenCASCADE, Mesa, or VTK installed. The training runtime requires only the standard PyTorch + PyArrow stack.
- **Pre-generation of Trajectories**:
  Preprocessing is run either on local workstations, cloud compute instances, or dedicated pre-processing/CPU nodes on the cluster.

---

## 2. Portability vs. Machine-Specific Infrastructure

When deploying to HPC systems like Jean Zay, software layers must be cleanly separated between what the supercomputer infrastructure provides versus what the repository owns:

```text
┌─────────────────────────────────────────────────────────────┐
│ Application Code: CADynamics (src/)                         │  ← Project Repository
├─────────────────────────────────────────────────────────────┤
│ Python Research Stack: PyTorch, PyArrow, Pandas             │  ← Project Managed (uv)
├─────────────────────────────────────────────────────────────┤
│ CUDA Runtime Libraries: libcufft, libcublas, libcudnn       │  ← Packaged in PyTorch Wheels
├─────────────────────────────────────────────────────────────┤
│ NVIDIA Host Kernel Driver (libcuda.so)                     │  ← HPC Infrastructure (Host OS)
├─────────────────────────────────────────────────────────────┤
│ Physical Accelerators (NVIDIA A100 / H100 SXM)             │  ← HPC Compute Hardware
└─────────────────────────────────────────────────────────────┘
```

### Layer Classification

1. **Host Infrastructure (HPC-Managed)**:
   - **Kernel Driver**: Provided by the HPC operating system. Never pin or package local workstation kernel drivers.
   - **CUDA Driver API (`libcuda.so`)**: Interfaced dynamically by PyTorch at runtime.
   - **InfiniBand / High-Speed Fabric**: Mellanox / OFED network stack managed via SLURM environment modules.

2. **Environment & Runtime (Project-Managed)**:
   - **PyTorch & CUDA Runtime**: Managed via project dependencies (`torch==2.4.1`). PyTorch wheels bundle the user-space CUDA runtime libraries (`libcufft`, `libcublas`, `libcudnn`, etc.), decoupling the project from cluster CUDA toolkit modules (`module load cuda/...`).
   - **Python Environment**: Isolated via Miniforge/Conda or Apptainer container as recommended by the computing facility.

---

## 3. Adapting to Jean Zay (IDRIS / GENCI)

### Compute Hardware Considerations
- Workstations typically feature NVIDIA T4, RTX, or consumer GPUs running CUDA 12.1.
- Jean Zay features NVIDIA V100 (32 GB) and NVIDIA A100 (80 GB) nodes, with specific multi-GPU SXM topology and Slurm job schedulers.
- The software environment defined in `environment.yml` and `pyproject.toml` is target-agnostic and will seamlessly utilize A100/H100 tensor cores without code modifications.

### Environment Strategy on Jean Zay
Two standard deployment models exist for Jean Zay:

1. **User-Space Conda + uv in `$WORK`**:
   - Install Miniconda/Miniforge on the persistent high-performance shared filesystem (`$WORK` or `$SCRATCH`).
   - Bootstrap using `./scripts/bootstrap_env.sh`.
   - Avoid installing environments in `$HOME` due to strict inode and quota limitations.

2. **Apptainer / Singularity Container (Recommended for Extreme Scale)**:
   - Build a container image from `environment.yml` and `pyproject.toml` / `uv.lock`.
   - Mount Jean Zay storage partitions (`$WORK`, `$SCRATCH`, `$DSDIR`) into the container.
   - Container execution eliminates file-system metadata overhead when loading thousands of Python files across multi-node distributed training jobs.

> [!NOTE]
> SLURM submission scripts will be authored separately once exact allocation project codes, partition queues (e.g. `gpu_p2`, `gpu_p5`), and time limits are confirmed with IDRIS documentation.
