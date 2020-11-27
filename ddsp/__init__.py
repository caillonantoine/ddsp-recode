import yaml
from .model import DDSP


def get_model(config="ddsp_config.yaml"):
    with open(config, "r") as config:
        config = yaml.safe_load(config)
    return DDSP(
        recurrent_args=config["recurrent_args"],
        harmonic_args=config["harmonic_args"],
        noise_args=config["noise_args"],
        scales=config["scales"],
    )
