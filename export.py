import torch
import yaml
from effortless_config import Config
from os import path, makedirs
from ddsp.model import DDSP
import soundfile as sf

torch.set_grad_enabled(False)

EXPORT_DIR = "export"
makedirs(EXPORT_DIR, exist_ok=True)


class args(Config):
    RUN = None
    CACHE = False


class ScriptDDSP(torch.nn.Module):
    def __init__(self, ddsp, mean_loudness, std_loudness):
        super().__init__()
        self.ddsp = ddsp
        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))

    def forward(self, pitch, loudness):
        loudness = (loudness - self.mean_loudness) / self.std_loudness
        return self.ddsp(pitch, loudness)


args.parse_args()

with open(path.join(args.RUN, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

ddsp = DDSP(**config["model"])

state = ddsp.state_dict()
pretrained = torch.load(path.join(args.RUN, "state.pth"), map_location="cpu")
state.update(pretrained)
ddsp.load_state_dict(state)

name = path.basename(path.normpath(args.RUN))

scripted_model = torch.jit.script(
    ScriptDDSP(
        ddsp,
        config["data"]["mean_loudness"],
        config["data"]["std_loudness"],
    ))
torch.jit.save(
    scripted_model,
    path.join(EXPORT_DIR, f"ddsp_{name}_pretrained.ts"),
)

impulse = ddsp.reverb.build_impulse().reshape(-1).numpy()
sf.write(
    path.join(EXPORT_DIR, f"ddsp_{name}_impulse.wav"),
    impulse,
    config["preprocess"]["sampling_rate"],
)

with open(
        path.join(EXPORT_DIR, f"ddsp_{name}_config.yaml"),
        "w",
) as config_out:
    yaml.safe_dump(config, config_out)