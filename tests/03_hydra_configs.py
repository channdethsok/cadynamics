"""Hydra configuration composition and schema consistency tests.

Verifies:
  1. Default root composition loads cleanly without missing config errors.
  2. All model and data configuration variants resolve without interpolation errors.
  3. Configuration dimensions strictly match Schema 0.3.0 constants.
  4. Dynamic CLI parameter overrides work as expected.
"""

from __future__ import annotations

import pytest

hydra = pytest.importorskip("hydra", reason="hydra-core is not installed in this environment")
omegaconf = pytest.importorskip("omegaconf", reason="omegaconf is not installed in this environment")

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.data.schema import EDGE_FEATURE_NAMES, FACE_FEATURE_NAMES, SCHEMA_VERSION


@pytest.fixture(autouse=True)
def clean_hydra() -> None:
    """Ensure GlobalHydra singleton is reset before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_default_config_composition() -> None:
    """Verify that root config.yaml composes default groups without errors."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config")
        assert cfg is not None
        assert cfg.data.name == "zero_to_cad_100k"
        assert cfg.model.name == "dummy"
        assert cfg.trainer.lr == 1e-4
        assert cfg.trainer.precision == "16-mixed"

        # Ensure fully resolved container contains no unresolved interpolations
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(resolved, dict)
        assert resolved["seed"] == 42


@pytest.mark.parametrize("data_group", ["zero_to_cad", "debug"])
@pytest.mark.parametrize("model_group", ["dummy", "brep_encoder"])
def test_all_config_combinations_resolve(data_group: str, model_group: str) -> None:
    """Test full cross-product of data and model configs with dynamic interpolation."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[f"data={data_group}", f"model={model_group}"],
        )
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(resolved, dict)

        # Invariant dimension check
        assert resolved["data"]["face_feature_dim"] == len(FACE_FEATURE_NAMES)
        assert resolved["data"]["edge_feature_dim"] == len(EDGE_FEATURE_NAMES)
        assert len(FACE_FEATURE_NAMES) == 32
        assert len(EDGE_FEATURE_NAMES) == 16

        # Invariant check for B-Rep encoder dynamic interpolation
        if model_group == "brep_encoder":
            assert resolved["model"]["face_feature_dim"] == resolved["data"]["face_feature_dim"]
            assert resolved["model"]["edge_feature_dim"] == resolved["data"]["edge_feature_dim"]
            assert resolved["model"]["hidden_dim"] > 0
            assert resolved["model"]["num_heads"] > 0


def test_cli_overrides() -> None:
    """Verify dynamic hyperparameter overrides via CLI-style override syntax."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=debug",
                "model=brep_encoder",
                "trainer.lr=5e-5",
                "data.batch_size=16",
                "seed=123",
            ],
        )
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["seed"] == 123
        assert resolved["trainer"]["lr"] == 5e-5
        assert resolved["data"]["batch_size"] == 16
        assert resolved["data"]["name"] == "debug"
        assert resolved["model"]["name"] == "brep_encoder"
        assert resolved["model"]["face_feature_dim"] == 32
        assert resolved["model"]["edge_feature_dim"] == 16
