from types import SimpleNamespace

import torch
import yaml

from ddsp.model import DDSP

with open("ddsp_config.yaml", "r") as config:
    config = yaml.safe_load(config)
    config = SimpleNamespace(**config)
    ddsp = DDSP(
        config.recurrent_args,
        config.harmonic_args,
        config.noise_args,
    ).cuda()

f0 = torch.randn(1, 100, 1).cuda()
lo = torch.randn(1, 100, 1).cuda()

y = ddsp(f0, lo)
print(y.shape)