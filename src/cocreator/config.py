"""Configuration management for cocreator.

Handles YAML config loading with environment variable substitution.
"""

import os
import re
from pathlib import Path

import yaml

from .schemas import AppConfig


def load_config(path: str) -> AppConfig:
    """
    Load YAML configuration file with ${ENV_VAR} substitution.

    Args:
        path: Path to YAML config file

    Returns:
        AppConfig: Validated configuration object

    Raises:
        ValueError: If YAML file not found or env var not set
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(raw_config)

    # Parse into AppConfig
    return AppConfig.model_validate(config)


def _substitute_env_vars(obj):
    """
    Recursively substitute ${ENV_VAR} in config values.
    """
    if isinstance(obj, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, obj)
        for var_name in matches:
            if var_name not in os.environ:
                raise ValueError(f"Environment variable not set: {var_name}")
            obj = obj.replace(f"${{{var_name}}}", os.environ[var_name])
        return obj
    elif isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    else:
        return obj