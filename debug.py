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

y, artifacts = ddsp(f0, lo)
y = y.reshape(-1).numpy()

Y = li.amplitude_to_db(li.feature.melspectrogram(y))
plt.imshow(Y, aspect="auto", origin="lower")

plt.colorbar()

plt.show()

plt.plot(artifacts["amp"][0].reshape(-1))
plt.show()

plt.imshow(artifacts["alphas"][0], aspect="auto")
plt.colorbar()
plt.show()

plt.plot(artifacts["impulse"].reshape(-1))
plt.show()

plt.plot(y.reshape(-1))
plt.show()
# %%
import torch
import torch.fft as fft
x = 2 * torch.rand(8192) - 1
x_f = fft.rfft(x, norm="backward")

y = fft.irfft(x_f, norm="backward")

print(x)
print(y)
