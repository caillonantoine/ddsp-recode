import torch
import yaml
from effortless_config import Config
from os import path
from ddsp.model import DDSP


def reshape_descriptor(x):
    if len(x.shape) == 1:
        x = x.reshape(1, -1, 1)
    else:
        x = x.reshape(x.shape[0], -1, 1)
    return x


class ScriptableModel(torch.nn.Module):
    def __init__(self, ddsp):
        super().__init__()
        self.ddsp = ddsp

    def forward(self, pitch, loudness):
        pitch = reshape_descriptor(pitch)
        loudness = reshape_descriptor(loudness)
        return self.ddsp(pitch, loudness)


class args(Config):
    RUN = None
    CACHE = False


args.parse_args()

with open(path.join(args.RUN, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

ddsp = DDSP(**config["model"])
state = ddsp.state_dict()
pretrained = torch.load(path.join(args.RUN, "state.pth"), map_location="cpu")
state.update(pretrained)
ddsp.load_state_dict(state)

name = path.basename(path.normpath(args.RUN))

scripted_model = torch.jit.script(ddsp)
torch.jit.save(scripted_model, f"ddsp_{name}_pretrained.ts")