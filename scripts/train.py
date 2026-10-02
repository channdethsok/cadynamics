"""Entrypoint for CADynamics training and experiment management using Hydra.

Usage Examples:
    # Run with default settings (dummy model + zero_to_cad data)
    uv run python scripts/train.py

    # Select the B-Rep encoder config and override learning rate
    uv run python scripts/train.py model=brep_encoder trainer.lr=5e-5

    # Run fast debug check
    uv run python scripts/train.py data=debug trainer.max_epochs=1
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate model based on the selected configuration."""
    model_name = cfg.model.name
    logger.info(f"Initializing model architecture: '{model_name}'")

    if model_name == "dummy":
        # Minimal linear model to verify pipeline flow without overhead
        return nn.Sequential(
            nn.Linear(cfg.model.input_dim, cfg.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.model.hidden_dim, cfg.model.output_dim),
        )
    elif model_name == "brep_encoder":
        logger.info(
            "B-Rep encoder architecture selected: "
            f"face_dim={cfg.model.face_feature_dim}, "
            f"edge_dim={cfg.model.edge_feature_dim}, "
            f"hidden_dim={cfg.model.hidden_dim}, "
            f"layers={cfg.model.num_layers}"
        )
        # Note: Implement actual B-Rep encoder in src/models/brep_encoder.py
        # and import here when ready!
        raise NotImplementedError(
            "B-Rep encoder is currently an idea/spec in configs/model/brep_encoder.yaml. "
            "Implement the module in src/models/brep_encoder.py when ready!"
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Log resolved configuration
    logger.info("=" * 60)
    logger.info("RESOLVED EXPERIMENT CONFIGURATION:")
    logger.info("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    logger.info("=" * 60)

    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    logger.info(f"Random seed set to {cfg.seed}")

    # Build model (supports placeholder / dummy while ideas develop)
    try:
        model = build_model(cfg)
        logger.info(f"Model successfully constructed:\n{model}")
    except NotImplementedError as e:
        logger.warning(f"Model placeholder notice: {e}")
        return

    logger.info("Ready for training loop execution.")


if __name__ == "__main__":
    main()
