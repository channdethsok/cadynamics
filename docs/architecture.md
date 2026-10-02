# CADynamics Architecture & System Design

> **Modeling State Transitions & Dynamics in Parametric Geometry**

---

## 1. Overview & Objectives

Traditional 3D generative pipelines operate on unstructured point clouds, voxel grids, or discrete triangle meshes. In contrast, mechanical engineering and industrial design rely fundamentally on **Boundary Representation (B-Rep)** and **parametric Computer-Aided Design (CAD)** construction histories (sketches, extrusions, cuts, fillets, chamfers).

**CADynamics** models the causal dynamics of parametric geometry:
$$\mathbf{z}_{t+1} \approx \operatorname{Predictor}(\mathbf{z}_t, A_t)$$
where $\mathbf{z}_t$ is the latent embedding of the current geometric/visual CAD state $S_t$, $A_t$ is a parameterized modeling operation, and $\mathbf{z}_{t+1}$ is the predicted latent representation of the updated state $S_{t+1}$.

```
                 ┌────────────────────────────────┐
                 │       Action A_t               │
                 │  (cmd_id, params, 2D profile)  │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Latent State z_t ├───►│ JEPA Predictor  ├───►│ Predicted z_{t+1}│
└──────────────────┘    └─────────────────┘    └────────┬─────────┘
         ▲                                              │ Loss:
         │                                              │ L2 / Cosine
┌────────┴─────────┐                           ┌────────┴─────────┐
│ Target Encoder   │                           │ Target Encoder   │
└────────┬─────────┘                           └────────┬─────────┘
         │                                              │
┌────────┴─────────┐                           ┌────────┴─────────┐
│     State S_t    │                           │   State S_{t+1}  │
│ (Images + B-Rep) │                           │ (Images + B-Rep) │
└──────────────────┘                           └──────────────────┘
```

---

## 2. Dual-Domain State Representation ($S_t$)

Intermediate states $S_t$ capture both **continuous visual projections** and **discrete topological boundary geometry**.

### 2.1 Visual Domain (Multiview Canonical Perspectives)
Rendered offscreen via OpenCASCADE / VTK at $224 \times 224$ resolution with orthographic/axonometric projection:
* **`iso`**: Axonometric isometric perspective ($\text{Azimuth} = 45^\circ, \text{Elevation} = 30^\circ$), showcasing 3D depth and volume.
* **`front`**: Orthographic front projection ($XZ$ plane).
* **`top`**: Orthographic top projection ($XY$ plane).
* **`right`**: Orthographic right projection ($YZ$ plane).

Stored as: `images: FloatTensor[4, 3, 224, 224]` (normalized to $[0, 1]$ or ImageNet standard).

### 2.2 Boundary Representation (B-Rep) Cell Complex
B-Rep geometry represents 3D solids via hierarchical topological cells: **Faces (2D)** bounded by **Edges (1D)** bounded by **Vertices (0D)**.

#### Faces (2D Geometry & Topology)
* **`faces_features`** (`FloatTensor[N_faces, 32]`):
  * **[0..7] Surface Type One-Hot**: Plane, Cylinder, Cone, Sphere, Torus, B-spline/Bezier, Revolution/Extrusion, Other.
  * **[8] Metric Area**: SymLog-scaled surface area.
  * **[9..11] Centroid**: ($c_x, c_y, c_z$) normalized via $\tanh(x / 100.0)$.
  * **[12..14] Midpoint Normal**: Oriented unit normal vector $\mathbf{n} \in \mathbb{R}^3$.
  * **[15..17] Bounding Box**: Dimensions $(\Delta x, \Delta y, \Delta z)$.
  * **[18..20] Topology**: Number of trimming wires, number of boundary edges, TopAbs orientation flag ($\pm 1.0$).
  * **[21..22] UV Spans**: Parametric domain ranges ($\Delta u, \Delta v$).
  * **[23..25] Inertia**: Principal moments of inertia ($I_{xx}, I_{yy}, I_{zz}$).
  * **[26..31] Geometric Invariants**: $\log(1 + \text{area})$, aspect ratios, bounding box diagonal, planarity/curvature flags.
* **`faces_uv`** (`FloatTensor[N_faces, 7, 16, 16]`):
  * Grid of $16 \times 16$ uniform UV domain evaluations containing $[x, y, z, n_x, n_y, n_z, \text{mask}]$.

#### Edges (1D Curve Geometry)
* **`edges_features`** (`FloatTensor[N_edges, 16]`):
  * **[0..5] Curve Type One-Hot**: Line, Circle, Ellipse, Conic (Parabola/Hyperbola), B-spline/Bezier, Other.
  * **[6] Arc Length**: SymLog-scaled curve length.
  * **[7..9] Midpoint**: Tanh-normalized 3D midpoint coordinate.
  * **[10..12] Unit Tangent**: Direction vector $\mathbf{t} \in \mathbb{R}^3$ at curve midpoint.
  * **[13..15] Flags & Convexity**: `is_closed`, `is_degenerated`, and dihedral angle convexity ($+1.0$ convex, $-1.0$ concave, $0.0$ smooth/laminar).
* **`edges_u`** (`FloatTensor[N_edges, 6, 16]`):
  * Discretized 1D curve samples across 16 uniform parameter intervals $u \in [u_{\min}, u_{\max}]$: $[x, y, z, t_x, t_y, t_z]$.

#### Topological Graph
* **`faces_adjacency_index`** (`LongTensor[2, E_adj]`):
  * Directed edge list defining face-to-face adjacency across shared topological boundaries.

---

## 3. Action Representation ($A_t$)

Parametric CAD operations alter state through discrete operator selection combined with continuous transformation parameters.

* **`cmd_id`** (`LongTensor[]`): Categorical command identifier mapped against `CANONICAL_COMMANDS` (`EXTRUDE`, `CUT`, `FILLET`, `CHAMFER`, `REVOLVE`, etc.).
* **`params`** (`FloatTensor[12]`): Continuous parameters scaled via SymLog:
  $$\operatorname{symlog}(x) = \operatorname{sign}(x) \cdot \ln(1 + |x|)$$
* **`param_mask`** (`BoolTensor[12]`): Binary mask indicating valid parameters for the specific operation.
* **`ref_kind`** (`LongTensor[]`): Reference target entity type (`REF_NONE`, `REF_WORKPLANE`, `REF_FACE`, `REF_EDGE`, `REF_VERTEX`).
* **`profile_points_2d`** (`FloatTensor[64, 2]`): 2D sketch profile discretized into 64 sampled points projected into the local workplane coordinate frame:
  $$\mathbf{p}_{\text{local}} = \mathbf{R}_{\text{plane}}^T (\mathbf{p}_{\text{world}} - \mathbf{o}_{\text{plane}})$$

---

## 4. End-to-End Pipeline & Lineage

```
[Hugging Face / Zero-to-CAD 100K]
                 │
                 │ Parquet table (STEP strings, CadQuery source code)
                 ▼
[scripts/02_preprocess_data.py]
  ├── CadQuery AST Tracer (captures intermediate states & parameters)
  ├── OpenCASCADE Core Extractor (computes 32-dim faces & 16-dim edges)
  └── Headless CadRenderer (renders 4 canonical views @ 224x224)
                 │
                 │ 1-to-1 Deterministic Sharding
                 ▼
[data/processed_data/<split>/<stem>.pt]
  ├── metadata (schema_version, lineage, vocabularies)
  └── records[] (trajectories with states[] and actions[])
                 │
                 │ High-throughput PyTorch I/O (300+ samples/sec)
                 ▼
[src/data/dataset.py: CADTransitionDataset]
  └── collate_transition_batch (variable graph disjoint union + batched images)
                 │
                 ▼
[Dynamics Model Training Loop]
```

### Deterministic Lineage Architecture
Every `.pt` shard directly mirrors the source `.parquet` shard by filename stem. Each trajectory record preserves `source_shard`, `source_row`, and `part_id`, allowing $O(1)$ reverse lookup back to the original CadQuery Python script, raw STEP solid, and tessellated mesh.
