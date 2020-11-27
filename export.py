import torch
from ddsp.model import ScriptableDDSP
from effortless_config import Config
from os import path
import yaml


class args(Config):
    RUN = None


args.parse_args()

with open(path.join(args.RUN, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

ddsp = ScriptableDDSP(
    recurrent_args=config["recurrent_args"],
    harmonic_args=config["harmonic_args"],
    noise_args=config["noise_args"],
    scales=config["scales"],
)

state = ddsp.state_dict()
pretrained = torch.load(path.join(args.RUN, "state.pth"), map_location="cpu")
state.update(pretrained)

ddsp.load_state_dict(state)
ddsp.eval()

name = path.basename(path.normpath(args.RUN))
torch.jit.save(torch.jit.script(ddsp), f"{name}.ts")
