# Documentation & Research Notes

Welcome to the **CADynamics** documentation directory. This folder contains architectural specifications, mathematical formulations, and engineering logbooks.

---

## 📚 Documents Overview

### 1. System Architecture
* **[`architecture.md`](architecture.md)**: Comprehensive design document detailing:
  * Dual-domain CAD representation: 4-view canonical projections (`images`) and B-Rep cell complexes (`faces_features`, `edges_features`, `faces_adjacency_index`).
  * Action representation: Discrete command vocabulary, SymLog-scaled parameters, and 2D sketch profile grounding.
  * Pipeline lineage: 1-to-1 deterministic Parquet-to-PT shard mapping with $O(1)$ reverse lookup.

---

### 2. Research & Engineering Logbooks (`docs/`)
Chronological devlogs capturing technical decisions, schema evolutions, and benchmarking results:

| Date & Log | Topic / Focus Area |
| :--- | :--- |
| **[`2026-09-30_13h.md`](research_logs/2026-09-30_13h.md)** | Initial transition-pair dataset concept & prototype PyG DataLoader. |
| **[`2026-09-30_15h.md`](research_logs/2026-09-30_15h.md)** | AST instrumentation, CadQuery runtime execution, and offline replay engine. |
| **[`2026-09-30_17h.md`](research_logs/2026-09-30_17h.md)** | Headless offscreen renderer setup, camera angles, and 4-view canonical renders. |
| **[`2026-09-30_21h.md`](research_logs/2026-09-30_21h.md)** | Schema 0.2.0: Plural state naming, 16-dim 1D edge curve features & 2D profile grounding. |
| **[`2026-09-30_23h.md`](research_logs/2026-09-30_23h.md)** | 1-to-1 Parquet-to-PT shard mapping, fault-tolerant preprocessing & bidirectional lineage. |
| **[`2026-10-01_08h.md`](research_logs/2026-10-01_08h.md)** | Intermediate STEP/STL export, debug architecture & visual timelapses. |
| **[`2026-10-01_22h.md`](research_logs/2026-10-01_22h.md)** | Phase 1 codebase audit, literature benchmarks & representation reconsiderations. |
| **[`2026-10-02_01h.md`](research_logs/2026-10-02_01h.md)** | Experiment configuration system (Hydra), architecture decoupling & pipeline verification. |

---

## 📝 Conventions for New Logs
When adding a new research log in `research_logs/`:
1. Use the format `YYYY-MM-DD_<time>h.md` (e.g. `2026-10-01_10h.md`).
2. Include sections for:
   * **Context & Motivation**: What problem or hypothesis is being tackled.
   * **Technical Decisions / Math**: Equations, schema changes, or algorithms.
   * **Verification & Results**: Test runs, metrics, or tensor shape validations.
