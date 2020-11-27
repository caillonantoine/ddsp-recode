import torch
from ddsp.model import ScriptableDDSP
from effortless_config import Config
import soundfile as sf
from os import path
import yaml


class args(Config):
    RUN = "runs/full_model"
    CACHE = False


args.parse_args()

with open(path.join(args.RUN, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

config["recurrent_args"].update({"cache": args.CACHE})

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
torch.jit.save(torch.jit.script(ddsp), f"{name}_script.ts")

second = torch.randn(1, 1, 2 * config["harmonic_args"]["sampling_rate"])
impulse = ddsp.reverb.get_impulse(second).cpu().detach().numpy().reshape(-1)

sf.write(
    f"{name}_impulse.wav",
    impulse,
    config["harmonic_args"]["sampling_rate"],
)
