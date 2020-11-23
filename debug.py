#%%

from types import SimpleNamespace

import torch

torch.set_grad_enabled(False)

import yaml

from ddsp.model import DDSP
import matplotlib.pyplot as plt
import librosa as li

with open("ddsp_config.yaml", "r") as config:
    config = yaml.safe_load(config)
    config = SimpleNamespace(**config)
    ddsp = DDSP(config.recurrent_args, config.harmonic_args, config.noise_args,
                config.training["scales"])

f0 = torch.linspace(100, 300, 100).reshape(1, 100, 1)
lo = torch.randn(1, 100, 1)

y = ddsp(f0, lo).reshape(-1).numpy()

Y = li.amplitude_to_db(li.feature.melspectrogram(y))
plt.imshow(Y, aspect="auto", origin="lower")
plt.colorbar()
# %%
