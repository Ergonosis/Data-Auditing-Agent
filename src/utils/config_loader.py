"""Configuration file loader with validation"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from .errors import ConfigurationError


def load_config(config_path: str = "config/rules.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file with validation.
    Automatically loads demo config if in demo mode.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration

    Raises:
        ConfigurationError: If file doesn't exist or invalid YAML
    """
    # Auto-select demo config if in demo mode
    if os.getenv("DEMO_MODE") == "true" or os.getenv("ENVIRONMENT") == "demo":
        if config_path == "config/rules.yaml":  # Only override if using default
            config_path = "config/rules_demo.yaml"

    try:
        config_file = Path(config_path)

        if not config_file.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required keys
        required_keys = ['version', 'rules', 'domain_configs', 'llm']
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ConfigurationError(f"Missing required configuration keys: {missing_keys}")

        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}")


def save_config(config_path: str, config: Dict[str, Any]) -> None:
    """
    Save configuration to YAML file

    Args:
        config_path: Path to configuration file
        config: Configuration dictionary

    Raises:
        ConfigurationError: If unable to write file
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    except Exception as e:
        raise ConfigurationError(f"Error saving configuration: {e}")


def get_domain_config(config: Dict[str, Any], domain: str = "default") -> Dict[str, Any]:
    """
    Get domain-specific configuration

    Args:
        config: Full configuration dictionary
        domain: Domain name (defaults to "default")

    Returns:
        Domain configuration dictionary
    """
    domain_configs = config.get('domain_configs', {})
    return domain_configs.get(domain, domain_configs.get('default', {}))
