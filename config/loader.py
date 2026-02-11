import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'streaming_config.yaml')


def load_streaming_config(path: str = None) -> dict:
    """Load streaming configuration YAML and return as dict.

    Args:
        path: Optional path to YAML file. If None, uses default in config/streaming_config.yaml
    Returns:
        Parsed config dict
    """
    p = path or CONFIG_PATH
    with open(p, 'r') as f:
        data = yaml.safe_load(f)
    return data
