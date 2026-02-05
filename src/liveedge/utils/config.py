"""Configuration loading and management utilities.

This module provides utilities for loading and managing configurations using Hydra/OmegaConf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration as a DictConfig object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return OmegaConf.load(config_path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations.

    Later configurations override earlier ones.

    Args:
        *configs: Configuration objects to merge.

    Returns:
        Merged configuration.
    """
    return OmegaConf.merge(*configs)


def to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert a DictConfig to a plain dictionary.

    Args:
        config: Configuration object.

    Returns:
        Plain dictionary representation.
    """
    return OmegaConf.to_container(config, resolve=True)


def validate_config(config: DictConfig, required_keys: list[str]) -> None:
    """Validate that required keys are present in configuration.

    Args:
        config: Configuration to validate.
        required_keys: List of required key paths (e.g., "data.sampling_rate").

    Raises:
        ValueError: If any required key is missing.
    """
    missing = []
    for key in required_keys:
        try:
            OmegaConf.select(config, key, throw_on_missing=True)
        except Exception:
            missing.append(key)

    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")


def get_nested(config: DictConfig, key: str, default: Any = None) -> Any:
    """Get a nested configuration value safely.

    Args:
        config: Configuration object.
        key: Dot-separated key path (e.g., "model.architecture.num_classes").
        default: Default value if key not found.

    Returns:
        Configuration value or default.
    """
    try:
        return OmegaConf.select(config, key, throw_on_missing=True)
    except Exception:
        return default


def save_config(config: DictConfig, path: str | Path) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration to save.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def print_config(config: DictConfig, resolve: bool = True) -> None:
    """Print configuration in a readable format.

    Args:
        config: Configuration to print.
        resolve: Whether to resolve interpolations.
    """
    print(OmegaConf.to_yaml(config, resolve=resolve))
