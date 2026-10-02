"""Command and reference vocabulary for Action-JEPA-CAD.

Defines:
  - Canonical CAD commands and aliases for solid-modifying operations
  - Reference entity types (Workplane, Face, Edge, Vertex)
  - Explicit versioning and mapping functions
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

VOCABULARY_VERSION: str = "0.1.0"

# Reference entity types
REF_NONE: int = 0
REF_WORKPLANE: int = 1
REF_FACE: int = 2
REF_EDGE: int = 3
REF_VERTEX: int = 4

REF_KIND_TO_NAME: Dict[int, str] = {
    REF_NONE: "REF_NONE",
    REF_WORKPLANE: "REF_WORKPLANE",
    REF_FACE: "REF_FACE",
    REF_EDGE: "REF_EDGE",
    REF_VERTEX: "REF_VERTEX",
}

NAME_TO_REF_KIND: Dict[str, int] = {
    name: kind for kind, name in REF_KIND_TO_NAME.items()
}

# Canonical CAD command classes
CANONICAL_COMMANDS: List[str] = [
    "EXTRUDE",    # 0: Linear extrusion from sketch
    "CUT",        # 1: Material removal / boolean difference with sketch profile
    "HOLE",       # 2: Cylindrical / counterbored / countersunk holes
    "CHAMFER",    # 3: Edge or face beveling
    "FILLET",     # 4: Edge or face rounding
    "REVOLVE",    # 5: Rotational extrusion around an axis
    "BOX",        # 6: 3D Cuboid primitive
    "CYLINDER",   # 7: 3D Cylinder primitive
    "SPHERE",     # 8: 3D Sphere primitive
    "LOFT",       # 9: Smooth transition through multiple profiles
    "SWEEP",      # 10: Profile swept along a guide path
    "SHELL",      # 11: Hollow solid with constant wall thickness
    "UNION",      # 12: Boolean union / fuse
    "INTERSECT",  # 13: Boolean intersection
    "OFFSET2D",   # 14: 2D offset operation creating a solid
    "OTHER",      # 15: Explicit unmapped, auxiliary, or composite solid operation
]

COMMAND_VOCAB: List[str] = CANONICAL_COMMANDS

CMD2ID: Dict[str, int] = {cmd: idx for idx, cmd in enumerate(CANONICAL_COMMANDS)}
ID2CMD: Dict[int, str] = {idx: cmd for idx, cmd in enumerate(CANONICAL_COMMANDS)}
NUM_COMMANDS: int = len(CANONICAL_COMMANDS)

# All known solid-modifying CadQuery methods to intercept
SOLID_COMMIT_OPS: Set[str] = {
    "extrude",
    "twistExtrude",
    "cut",
    "cutBlind",
    "cutThruAll",
    "holeThruAll",
    "hole",
    "cboreHole",
    "cskHole",
    "csk_face_hole",
    "chamfer",
    "fillet",
    "revolve",
    "box",
    "sphere",
    "cylinder",
    "wedge",
    "loft",
    "sweep",
    "shell",
    "offset2D",
    "union",
    "fuse",
    "intersect",
    "split",
    "mirror",
}

# Alias mapping from runtime CadQuery method names to canonical command
OP_TO_COMMAND: Dict[str, str] = {
    # Extrusions
    "extrude": "EXTRUDE",
    "twistextrude": "EXTRUDE",
    # Cuts
    "cut": "CUT",
    "cutblind": "CUT",
    "cutthruall": "CUT",
    "holethruall": "CUT",
    "split": "CUT",
    # Holes
    "hole": "HOLE",
    "cborehole": "HOLE",
    "cskhole": "HOLE",
    "csk_face_hole": "HOLE",
    # Modifications
    "chamfer": "CHAMFER",
    "fillet": "FILLET",
    # Revolutions
    "revolve": "REVOLVE",
    # Primitives
    "box": "BOX",
    "wedge": "BOX",
    "cylinder": "CYLINDER",
    "sphere": "SPHERE",
    # Advanced Swept
    "loft": "LOFT",
    "sweep": "SWEEP",
    "shell": "SHELL",
    "offset2d": "OFFSET2D",
    # Booleans
    "union": "UNION",
    "fuse": "UNION",
    "intersect": "INTERSECT",
    # Symmetries & Auxiliary
    "mirror": "OTHER",
    "other": "OTHER",
}


def normalize_op_name(op_name: str) -> str:
    """Return lowercase stripped operation name."""
    return op_name.strip().lower() if op_name else ""


def is_solid_commit_op(op_name: str) -> bool:
    """Check if a CadQuery operation is a registered solid-modifying command."""
    cleaned = normalize_op_name(op_name)
    return any(cleaned == cand.lower() for cand in SOLID_COMMIT_OPS)


def get_canonical_command(op_name: str) -> Optional[str]:
    """Map a CadQuery operation name to its canonical command, or None if unmapped."""
    cleaned = normalize_op_name(op_name)
    return OP_TO_COMMAND.get(cleaned, None)


def get_command_id(op_name: str) -> int:
    """Get discrete command ID for an operation name (defaults to OTHER if unmapped)."""
    canonical = get_canonical_command(op_name)
    if canonical is None or canonical not in CMD2ID:
        return CMD2ID["OTHER"]
    return CMD2ID[canonical]

