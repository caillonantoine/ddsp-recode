#%%
import torch
from ddsp.model import DDSP
import matplotlib.pyplot as plt
import librosa as li
import yaml
torch.set_grad_enabled(False)


def specshow(y):
    Y = li.feature.melspectrogram(y.numpy().reshape(-1))
    Y = li.amplitude_to_db(Y)
    plt.imshow(Y, aspect="auto", origin="lower")
    plt.show()


with open("ddsp_config.yaml", "r") as config:
    config = yaml.safe_load(config)

ddsp = DDSP(
    recurrent_args=config["recurrent_args"],
    harmonic_args=config["harmonic_args"],
    noise_args=config["noise_args"],
    scales=[2048, 1024, 512, 256, 128, 64],
)

pitch = torch.linspace(100, 200, 100).reshape(1, 1, -1)
loudness = torch.linspace(100, 200, 100).reshape(1, 1, -1)

y = ddsp(pitch, loudness)
specshow(y)