# CADynamics: Parametric CAD Construction Dynamics

CADynamics models the causal dynamics of parametric Computer-Aided Design (CAD) operations. Given an intermediate CAD state $S_t$ and a modeling action $A_t$ (extrude, cut, fillet, chamfer, etc.), the framework represents and predicts the state transition to $S_{t+1}$ across both visual rendering and boundary representation (B-Rep) topological domains.

---

## 📁 Repository Layout

```text
cadynamics/
├── environment.yml                 # Portable Conda environment (CAD/native stack)
├── pyproject.toml                  # Python package dependencies & metadata
├── uv.lock                         # Exact locked resolution for Python/ML stack
├── README.md                       # Project documentation and quickstart
├── .gitignore                      # Clean ignore rules (data, logs, renders)
│
├── env/                            # Diagnostic environment snapshots
│   ├── conda-explicit.txt          # Machine-specific explicit package snapshot
│   ├── conda-full.yml              # Complete Conda environment export
│   ├── conda-history.yml           # Explicitly requested Conda specs
│   ├── python-freeze.txt           # Python package freeze snapshot
│   └── environment-report.txt      # Hardware & runtime diagnostic report
│
├── src/
│   └── data/                       # Core data extraction, schema, and loading
│       ├── __init__.py             # Public API exports
│       ├── dataset.py              # High-throughput PyTorch CADTransitionDataset
│       ├── schema.py               # Schema 0.3.0 definitions & topological invariants
│       ├── vocabulary.py           # Command & entity vocabulary mappings
│       ├── occ_extractor.py        # OpenCASCADE B-Rep face/edge feature extraction
│       ├── action_extractor.py     # Parameter, reference, & 2D profile extraction
│       ├── renderer.py             # Headless offscreen 4-view CAD renderer
│       └── instrumentation.py      # CadQuery runtime AST tracer
│   └── models/                     # Predictive dynamics models and encoders
│
├── scripts/
│   ├── env/                              # Dedicated environment lifecycle scripts
│   │   ├── bootstrap.sh                  # One-command reproducible environment bootstrap
│   │   ├── freeze.sh                     # Idempotent diagnostic snapshot generator
│   │   └── verify.py                     # Environment & CAD kernel verification script
│   ├── 01_download_data.py               # Dataset downloader from Hugging Face
│   └── 02_preprocess_data.py             # Offline AST replay & state serialization
│
├── tests/
│   ├── __init__.py
│   ├── 00_preprocessing.py             # Raw Parquet ingestion -> Shard serialization -> DataLoader E2E test
│   ├── 01_dataloader.py                # DataLoader benchmark and schema verification
│   └── 02_visual_inspection.py         # 3D STEP/STL export, 4-view PNGs, and GIF timelapses
│
├── notebooks/
│   ├── 01_explore_transition_dataset.ipynb
│   └── 01_verify_preprocessed_shards.ipynb
│
├── docs/
│   ├── README.md                   # Documentation index & research log navigation
│   ├── environment.md              # Environment & reproducibility architecture
│   ├── hpc.md                      # HPC / Jean Zay deployment guide
│   ├── architecture.md             # System design, representations, & pipeline spec
│   └── research_logs/              # Detailed engineering & research logbooks
└── data/                           # (Git-ignored) Raw & preprocessed datasets
    ├── zero_to_cad_100k/           # Downloaded raw Parquet shards
    └── processed_data/             # Serialized Schema 0.3.0 PyTorch .pt shards
```

---

## 📐 Data Representation (Schema 0.3.0)

Each sample yields a transition tuple $(S_t, A_t, S_{t+1})$:

### 1. State Representation ($S_t$, $S_{t+1}$)
- **4-View Canonical Images (`images`)**: `FloatTensor[4, 3, 224, 224]`
  - Standard perspectives: `iso` (axonometric isometric), `front`, `top`, `right`.
- **B-Rep Face Features (`faces_features`)**: `FloatTensor[N_faces, 32]`
  - 8-dim surface classification one-hot (`is_plane`, `is_cylinder`, `is_cone`, `is_sphere`, etc.)
  - Metric area, centroid, normal vector, oriented bounding box, inertia matrix diagonal, topological loops.
- **B-Rep Face UV Grid (`faces_uv`)**: `FloatTensor[N_faces, 7, 16, 16]`
  - Regular $16 \times 16$ UV surface parameter sampling ($x, y, z, n_x, n_y, n_z, \text{mask}$).
- **B-Rep Edge Features (`edges_features`)**: `FloatTensor[N_edges, 16]`
  - 6-dim curve type classification (`is_line`, `is_circle`, `is_ellipse`, `is_bspline`, etc.)
  - SymLog arc length, tanh-normalized midpoint, unit tangent vector, closed/degenerated flags, and dihedral convexity.
- **B-Rep Edge 1D Curve Samples (`edges_u`)**: `FloatTensor[N_edges, 6, 16]`
  - Uniform parameter samples along 1D boundary curve: ($x, y, z, t_x, t_y, t_z$).
- **Face Adjacency Index (`faces_adjacency_index`)**: `LongTensor[2, E_adj]`
  - Directed face-to-face topological adjacency across shared B-Rep edges.

### 2. Action Representation ($A_t$)
- **Command ID (`cmd_id`)**: `LongTensor[]` (Index in canonical vocabulary: `EXTRUDE`, `CUT`, `FILLET`, etc.)
- **Continuous Parameters (`params`)**: `FloatTensor[12]` (SymLog-scaled numeric parameters)
- **Parameter Mask (`param_mask`)**: `BoolTensor[12]` (Active/valid mask for parameter vector)
- **Sketch Profile (`profile_points_2d`)**: `FloatTensor[64, 2]` (Discretized 2D sketch profile points in local workplane coordinates)

---

## 🚀 Quickstart

### 1. Downloading Raw Data
```bash
python scripts/01_download_data.py val
```

### 2. Preprocessing Intermediate States
Run offline extraction from downloaded Zero-to-CAD Parquet shards:
```bash
python scripts/02_preprocess_data.py \
    --input-dir data/zero_to_cad_100k \
    --output-dir data/processed_data \
    --split val \
    --parts-per-shard 50
```

### 3. Training with PyTorch DataLoader
Load preprocessed shards with high throughput and batched graph collation:
```python
from torch.utils.data import DataLoader
from src.data import CADTransitionDataset, collate_transition_batch

# Initialize dataset
dataset = CADTransitionDataset(
    root_dir="data/processed_data",
    split="val",
    normalize_images=True,
    cache_shards_in_memory=True,  # Set True for maximum training throughput
)

# PyTorch DataLoader with graph collation
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_transition_batch,
)

for batch in loader:
    curr_images = batch["current_state"]["images"]          # [16, 4, 3, 224, 224]
    curr_faces  = batch["current_state"]["faces_features"]   # [Total_faces, 32]
    batch_faces = batch["current_state"]["batch_faces"]      # [Total_faces]
    actions     = batch["action"]["params"]                 # [16, 12]
    cmd_ids     = batch["action"]["cmd_id"]                 # [16]
    # Forward pass through model ...
```

### 4. Running Verification & Benchmarks
Run the automated test suite:
```bash
# Run all tests (preprocessing, dataloading, visual debugging)
pytest tests/

# Or run the dataloader benchmark directly
python tests/01_dataloader.py --num-workers 2 --batch-size 8
```

---

## 🛠 Environment & Reproducibility

CADynamics utilizes a hybrid environment strategy: Conda manages the native CAD stack (`cadquery`, `ocp`, `mesalib`, `pyvista`), while `uv` manages the Python ML/research stack (`torch`, `torchvision`, `pyarrow`, `pandas`).

### 1. Bootstrap on a Fresh Machine
```bash
git clone <repo-url>
cd CADynamics

./scripts/env/bootstrap.sh

conda activate cadyn-env
python scripts/env/verify.py
```

### 2. Verify Environment
Run the automated environment and CAD kernel verification script at any time:
```bash
conda activate cadyn-env
python scripts/env/verify.py
```

### 3. Freeze Environment Snapshot
Capture an idempotent diagnostic snapshot of the active environment without mutating packages:
```bash
conda activate cadyn-env
./scripts/env/freeze.sh
```

For complete technical documentation on the hybrid dependency architecture and HPC cluster deployment:
- [Environment Architecture Guide](docs/environment.md)
- [HPC & Jean Zay Transition Strategy](docs/hpc.md)
