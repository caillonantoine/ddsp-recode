#%%

from types import SimpleNamespace

import torch
import yaml

from ddsp.model import DDSP
import matplotlib.pyplot as plt

with open("ddsp_config.yaml", "r") as config:
    config = yaml.safe_load(config)
    config = SimpleNamespace(**config)
    ddsp = DDSP(config.recurrent_args, config.harmonic_args, config.noise_args,
                config.training["scales"])

f0 = torch.randn(1, 100, 1)
lo = torch.randn(1, 100, 1)

hidden = ddsp.recurrent_block([f0, lo])
hidden = torch.randn_like(hidden)

noise = ddsp.noise(hidden)

plt.plot(noise.reshape(-1).detach().cpu().numpy())
# %%
