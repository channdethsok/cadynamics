"""PyTorch Dataset and DataLoader collation for preprocessed CADynamics shards.

This module provides:
  - CADTransitionDataset: PyTorch Dataset yielding Schema 0.3.0 transition pairs (S_t, A_t, S_{t+1})
  - ZeroToCADTransitionDataset: Backwards-compatible alias for CADTransitionDataset
  - collate_transition_batch: Batched collation supporting both fixed-size tensors (images, actions)
    and variable-sized graph representations (faces, edges, adjacency) with PyG compatibility.
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class CADTransitionDataset(Dataset):
    """High-throughput PyTorch Dataset for preprocessed CADynamics transition shards.

    Each sample corresponds to a transition tuple:
        S_t  (current state)
        A_t  (action)
        S_{t+1} (next state)

    Schema 0.3.0 entities:
      - Multiview renders: 4 canonical perspectives ('iso', 'front', 'top', 'right')
      - B-Rep Face features: FloatTensor[N_faces, 32]
      - B-Rep Face UV grids: FloatTensor[N_faces, 7, 16, 16]
      - B-Rep Edge features: FloatTensor[N_edges, 16]
      - B-Rep Edge 1D curve samples: FloatTensor[N_edges, 6, 16]
      - Face adjacency graph: LongTensor[2, E_adj]
      - Action command ID, 12 continuous params, boolean param mask, and 2D sketch profile
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/processed_data",
        split: str = "train",
        normalize_images: bool = True,
        use_imagenet_norm: bool = False,
        cache_shards_in_memory: bool = False,
        max_samples: Optional[int] = None,
        max_cached_shards: int = 3,
        step_horizon: int = 1,
    ) -> None:
        """Initialize CADTransitionDataset.

        Args:
            root_dir: Directory containing preprocessed .pt shards (e.g. data/processed_data).
            split: Dataset split ('train', 'val', 'validation', 'test', 'all').
            normalize_images: If True, scales uint8 images to float32 [0.0, 1.0].
            use_imagenet_norm: If True and normalize_images is True, applies ImageNet mean/std.
            cache_shards_in_memory: If True, caches all loaded shard dicts in memory.
            max_samples: Limit total transitions loaded (useful for debugging/fast tests).
            max_cached_shards: Maximum open shards kept in worker process RAM LRU cache (default: 3).
            step_horizon: Step distance between current state S_t and target state S_{t+H} (default: 1).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = "val" if split == "validation" else split
        self.normalize_images = normalize_images
        self.use_imagenet_norm = use_imagenet_norm
        self.cache_shards_in_memory = cache_shards_in_memory
        self.max_cached_shards = max_cached_shards
        self.step_horizon = max(1, int(step_horizon))
        self.cached_shards: Dict[Path, Dict[str, Any]] = {}
        self._lru_cache: collections.OrderedDict[Path, Dict[str, Any]] = collections.OrderedDict()

        # 1. Discover all shard files
        self.shard_paths = self._find_shards()
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No preprocessed .pt shards found under '{self.root_dir}' for split '{self.split}'."
            )

        # 2. Build index of transitions: list of (shard_path, record_idx, step_t)
        self.index: List[Tuple[Path, int, int]] = self._build_transition_index(max_samples)
        logger.info(
            f"CADTransitionDataset (split='{self.split}', horizon={self.step_horizon}) ready with {len(self.index)} "
            f"transitions across {len(self.shard_paths)} shards."
        )

    def _find_shards(self) -> List[Path]:
        """Locate shard files matching split."""
        # Try split-specific subdirectory first
        split_dir = self.root_dir / self.split
        if split_dir.is_dir():
            shards = sorted(list(split_dir.glob("*.pt")))
            if shards:
                return shards

        # Search recursively under root_dir
        if self.split == "all":
            shards = sorted(list(self.root_dir.rglob("*.pt")))
        else:
            shards = sorted(list(self.root_dir.glob(f"**/{self.split}/**/*.pt")))
            if not shards:
                # Fallback to direct *.pt if under root
                shards = sorted(list(self.root_dir.glob("*.pt")))
        return shards

    def _build_transition_index(
        self, max_samples: Optional[int]
    ) -> List[Tuple[Path, int, int]]:
        """Index transitions across shards without keeping all tensors in memory."""
        index: List[Tuple[Path, int, int]] = []

        for shard_path in self.shard_paths:
            try:
                # Load shard header / records metadata
                shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
                if self.cache_shards_in_memory:
                    self.cached_shards[shard_path] = shard_data

                records = shard_data.get("records", [])
                for r_idx, rec in enumerate(records):
                    num_steps = rec.get("num_steps", len(rec.get("actions", [])))
                    # Multi-step transition sequences connecting S_t -> S_{t + step_horizon}
                    for t in range(num_steps - self.step_horizon + 1):
                        index.append((shard_path, r_idx, t))
                        if max_samples is not None and len(index) >= max_samples:
                            return index
            except Exception as e:
                logger.warning(f"Error indexing shard {shard_path}: {e}")

        return index

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, shard_path: Path) -> Dict[str, Any]:
        """Load shard using worker LRU cache or disk."""
        if self.cache_shards_in_memory and shard_path in self.cached_shards:
            return self.cached_shards[shard_path]

        if self.max_cached_shards > 0:
            if shard_path in self._lru_cache:
                # Cache hit: touch and move to most recently used
                self._lru_cache.move_to_end(shard_path)
                return self._lru_cache[shard_path]

            # Cache miss: load from disk and insert
            shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
            self._lru_cache[shard_path] = shard_data
            if len(self._lru_cache) > self.max_cached_shards:
                # Evict oldest entry (least recently used)
                self._lru_cache.popitem(last=False)
            return shard_data

        return torch.load(shard_path, map_location="cpu", weights_only=False)

    def _process_images(self, raw_images: torch.Tensor) -> torch.Tensor:
        """Process 4-view images tensor [4, 3, 224, 224]."""
        if not self.normalize_images:
            return raw_images

        # uint8 [0..255] -> float32 [0.0..1.0]
        img = raw_images.to(torch.float32) / 255.0

        if self.use_imagenet_norm:
            # Broadcast across [4, 3, 224, 224]
            img = (img - IMAGENET_MEAN) / IMAGENET_STD

        return img

    def _format_state(self, state_raw: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw state dictionary into sanitized, cloned tensors."""
        E_adj = state_raw["faces_adjacency_index"].shape[1]
        n_faces = state_raw.get("num_faces", state_raw["faces_features"].shape[0])
        n_edges = state_raw.get("num_edges", state_raw["edges_features"].shape[0])

        return {
            "faces_features": state_raw["faces_features"].clone(),
            "faces_uv": state_raw["faces_uv"].clone(),
            "edges_features": state_raw["edges_features"].clone(),
            "edges_features_directed": state_raw.get(
                "edges_features_directed",
                torch.zeros((E_adj, 16), dtype=torch.float32),
            ).clone(),
            "edges_u": state_raw["edges_u"].clone(),
            "faces_adjacency_index": state_raw["faces_adjacency_index"].clone(),
            "faces_adjacency_edge_indices": state_raw.get(
                "faces_adjacency_edge_indices",
                torch.zeros(E_adj, dtype=torch.long),
            ).clone(),
            "face_edge_index": state_raw.get(
                "face_edge_index",
                torch.zeros((2, 0), dtype=torch.long),
            ).clone(),
            "reverse_edge_indices": state_raw.get(
                "reverse_edge_indices",
                torch.zeros(E_adj, dtype=torch.long),
            ).clone(),
            "images": self._process_images(state_raw["images"]),
            "bbox": state_raw.get("bbox", torch.zeros(6, dtype=torch.float32)).clone(),
            "num_faces": n_faces,
            "num_edges": n_edges,
        }

    def _format_action(
        self, action_raw: Dict[str, Any], n_faces: int, n_edges: int
    ) -> Dict[str, Any]:
        """Format raw action dictionary into standardized tensors."""
        profile_pts = torch.zeros((64, 2), dtype=torch.float32)
        if isinstance(action_raw.get("profile"), dict):
            pts = action_raw["profile"].get("points_2d")
            if isinstance(pts, torch.Tensor) and pts.shape == torch.Size([64, 2]):
                profile_pts = pts.clone()

        sel_faces = action_raw.get("selected_faces_mask")
        sel_edges = action_raw.get("selected_edges_mask")
        if sel_faces is None or not isinstance(sel_faces, torch.Tensor) or sel_faces.shape != (n_faces,):
            sel_faces = torch.zeros(n_faces, dtype=torch.bool)
        if sel_edges is None or not isinstance(sel_edges, torch.Tensor) or sel_edges.shape != (n_edges,):
            sel_edges = torch.zeros(n_edges, dtype=torch.bool)

        return {
            "cmd_id": torch.tensor(action_raw["cmd_id"], dtype=torch.long),
            "params": action_raw["params"].clone(),
            "param_mask": action_raw["param_mask"].clone(),
            "ref_kind": torch.tensor(action_raw.get("ref_kind", 0), dtype=torch.long),
            "selected_faces_mask": sel_faces.clone(),
            "selected_edges_mask": sel_edges.clone(),
            "profile_points_2d": profile_pts,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        shard_path, record_idx, t = self.index[idx]
        shard_data = self._load_shard(shard_path)
        record = shard_data["records"][record_idx]

        curr_state_raw = record["states"][t]
        target_state_raw = record["states"][t + self.step_horizon]

        current_state = self._format_state(curr_state_raw)
        next_state = self._format_state(target_state_raw)

        # Sequence of actions over horizon [t, t + step_horizon)
        actions_list: List[Dict[str, Any]] = []
        for step_offset in range(self.step_horizon):
            act_raw = record["actions"][t + step_offset]
            ref_state = record["states"][t + step_offset]
            nf = ref_state.get("num_faces", ref_state["faces_features"].shape[0])
            ne = ref_state.get("num_edges", ref_state["edges_features"].shape[0])
            actions_list.append(self._format_action(act_raw, nf, ne))

        metadata = {
            "part_id": record.get("part_id", ""),
            "step": t,
            "step_horizon": self.step_horizon,
            "target_step": t + self.step_horizon,
            "num_steps": record.get("num_steps", len(record["actions"])),
            "source_shard": shard_path.name,
        }

        return {
            "current_state": current_state,
            "action": actions_list[0],
            "actions": actions_list,
            "next_state": next_state,
            "metadata": metadata,
        }


# Backwards compatibility alias
ZeroToCADTransitionDataset = CADTransitionDataset


def _collate_state_graphs(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of state dictionaries into batched graph and image tensors."""
    B = len(states)

    # 1. Fixed-size tensors: stack directly
    images = torch.stack([s["images"] for s in states], dim=0)  # [B, 4, 3, 224, 224]
    bboxes = torch.stack([s["bbox"] for s in states], dim=0)    # [B, 6]
    num_faces_t = torch.tensor([s["num_faces"] for s in states], dtype=torch.long)
    num_edges_t = torch.tensor([s["num_edges"] for s in states], dtype=torch.long)

    # 2. Variable-sized geometry: concatenate nodes and construct batch index pointers
    faces_features_list = [s["faces_features"] for s in states]
    faces_uv_list = [s["faces_uv"] for s in states]
    edges_features_list = [s["edges_features"] for s in states]
    edges_features_dir_list = [s["edges_features_directed"] for s in states]
    edges_u_list = [s["edges_u"] for s in states]

    faces_features = torch.cat(faces_features_list, dim=0)  # [Total_faces, 32]
    faces_uv = torch.cat(faces_uv_list, dim=0)              # [Total_faces, 7, 16, 16]
    edges_features = torch.cat(edges_features_list, dim=0)  # [Total_edges, 16]
    edges_features_directed = torch.cat(edges_features_dir_list, dim=0) if edges_features_dir_list else torch.zeros((0, 16), dtype=torch.float32)
    edges_u = torch.cat(edges_u_list, dim=0)                # [Total_edges, 6, 16]

    # Batch vectors indicating which graph sample each face/edge belongs to
    batch_faces = torch.repeat_interleave(torch.arange(B, dtype=torch.long), num_faces_t)
    batch_edges = torch.repeat_interleave(torch.arange(B, dtype=torch.long), num_edges_t)

    # 3. Disjoint union of adjacency indices with cumulative face and edge offsets
    shifted_adj_list: List[torch.Tensor] = []
    shifted_adj_edge_indices: List[torch.Tensor] = []
    shifted_face_edge_list: List[torch.Tensor] = []
    face_offset = 0
    edge_offset = 0

    for s in states:
        adj = s["faces_adjacency_index"]  # [2, E_adj]
        if adj.shape[1] > 0:
            shifted_adj_list.append(adj + face_offset)
            if "faces_adjacency_edge_indices" in s and s["faces_adjacency_edge_indices"].numel() > 0:
                shifted_adj_edge_indices.append(s["faces_adjacency_edge_indices"] + edge_offset)

        if "face_edge_index" in s and s["face_edge_index"].shape[1] > 0:
            fe = s["face_edge_index"]
            shifted_face_edge_list.append(torch.stack([fe[0] + face_offset, fe[1] + edge_offset], dim=0))

        face_offset += s["num_faces"]
        edge_offset += s["num_edges"]

    faces_adjacency_index = torch.cat(shifted_adj_list, dim=1) if shifted_adj_list else torch.zeros((2, 0), dtype=torch.long)
    faces_adjacency_edge_indices = torch.cat(shifted_adj_edge_indices, dim=0) if shifted_adj_edge_indices else torch.zeros(0, dtype=torch.long)
    face_edge_index = torch.cat(shifted_face_edge_list, dim=1) if shifted_face_edge_list else torch.zeros((2, 0), dtype=torch.long)

    return {
        "images": images,
        "bbox": bboxes,
        "num_faces": num_faces_t,
        "num_edges": num_edges_t,
        "faces_features": faces_features,
        "faces_uv": faces_uv,
        "batch_faces": batch_faces,
        "edges_features": edges_features,
        "edges_features_directed": edges_features_directed,
        "edges_u": edges_u,
        "batch_edges": batch_edges,
        "faces_adjacency_index": faces_adjacency_index,
        "faces_adjacency_edge_indices": faces_adjacency_edge_indices,
        "face_edge_index": face_edge_index,
    }


def collate_transition_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collation function for PyTorch DataLoader.

    Properly handles:
      - Variable-sized B-Rep face/edge graphs (via disjoint union & batch pointers)
      - Fixed-sized 4-view canonical images [B, 4, 3, 224, 224]
      - Action selection masks aligned with batch_faces and batch_edges
      - Discrete and continuous action parameters
    """
    curr_states = [sample["current_state"] for sample in batch]
    actions = [sample["action"] for sample in batch]
    next_states = [sample["next_state"] for sample in batch]
    metadatas = [sample["metadata"] for sample in batch]

    collated_curr = _collate_state_graphs(curr_states)
    collated_next = _collate_state_graphs(next_states)

    collated_action = {
        "cmd_id": torch.stack([a["cmd_id"] for a in actions], dim=0),                    # [B]
        "params": torch.stack([a["params"] for a in actions], dim=0),                    # [B, 12]
        "param_mask": torch.stack([a["param_mask"] for a in actions], dim=0),            # [B, 12]
        "ref_kind": torch.stack([a["ref_kind"] for a in actions], dim=0),                # [B]
        "selected_faces_mask": torch.cat([a["selected_faces_mask"] for a in actions], dim=0), # [Total_faces]
        "selected_edges_mask": torch.cat([a["selected_edges_mask"] for a in actions], dim=0), # [Total_edges]
        "profile_points_2d": torch.stack([a["profile_points_2d"] for a in actions], dim=0), # [B, 64, 2]
    }

    # Metric bounding box delta
    bbox_delta = collated_next["bbox"] - collated_curr["bbox"]  # [B, 6]

    batch_dict = {
        "current_state": collated_curr,
        "action": collated_action,
        "next_state": collated_next,
        "metadata": metadatas,
        "bbox_delta": bbox_delta,
    }

    # Support batched action sequences when step_horizon > 1
    if "actions" in batch[0] and len(batch[0]["actions"]) > 1:
        action_seqs = [sample["actions"] for sample in batch]
        batch_dict["action_sequence"] = {
            "cmd_id": torch.stack([torch.stack([a["cmd_id"] for a in seq], dim=0) for seq in action_seqs], dim=0),
            "params": torch.stack([torch.stack([a["params"] for a in seq], dim=0) for seq in action_seqs], dim=0),
            "param_mask": torch.stack([torch.stack([a["param_mask"] for a in seq], dim=0) for seq in action_seqs], dim=0),
            "ref_kind": torch.stack([torch.stack([a["ref_kind"] for a in seq], dim=0) for seq in action_seqs], dim=0),
            "profile_points_2d": torch.stack([torch.stack([a["profile_points_2d"] for a in seq], dim=0) for seq in action_seqs], dim=0),
        }

    return batch_dict
