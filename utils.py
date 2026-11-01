from pathlib import Path
import yaml

def load_config(config_path: str = "config.yaml") -> dict:
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"No hay archivo de configuración: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config
