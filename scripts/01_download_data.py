"""Download one Zero-to-CAD-100K split from Hugging Face.

Examples
--------
# Metadata and the training Parquet shards only.
python scripts/01_download_data.py train

# Download all three splits (large; includes images, STL, and STEP files).
python scripts/01_download_data.py all

The files are stored under ``data/zero_to_cad_100k`` by default.  This script
only downloads files; it never executes the CadQuery programs in the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ID = "ADSKAILab/Zero-To-CAD-100k"
DEFAULT_DATA_DIR = Path("data/zero_to_cad_100k")


def patterns_for_split(split: str) -> list[str] | None:
    """Return Hugging Face file patterns for a requested split."""
    if split == "all":
        return None
    # The public dataset split is named validation, but the repository
    # stores its Parquet shards in data/val.
    directory = {"validation": "val"}.get(split, split)
    return ["README.md", f"data/{directory}/**"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Zero-to-CAD-100K files from Hugging Face."
    )
    parser.add_argument(
        "split",
        choices=("train", "validation", "test", "all"),
        help="Split to download. 'all' downloads the complete dataset and is large.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Destination directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision / commit (default: main). Record a commit for reproducibility.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "Missing dependency. Install it with:\n"
            "  pip install -U huggingface_hub hf_xet\n"
        ) from error

    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Repository:  {REPO_ID}")
    print(f"Revision:    {args.revision}")
    print(f"Split:       {args.split}")
    print(f"Destination: {args.data_dir.resolve()}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.data_dir,
        allow_patterns=patterns_for_split(args.split),
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
