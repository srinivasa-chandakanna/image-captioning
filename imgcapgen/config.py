# imgcapgen/config/config.py
import yaml

class DotConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                v = DotConfig(v)
            setattr(self, k, v)

def load_config(path="configs/flickr8k.yaml"):
    """
    Loads a YAML config file into a DotConfig object.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DotConfig(data)
