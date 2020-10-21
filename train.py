import numpy as np
import crepe
from udls import SimpleDataset
import librosa as li
import yaml
from types import SimpleNamespace
import torch
from os import path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ddsp.model import DDSP
import matplotlib.pyplot as plt

with open("ddsp_config.yaml", "r") as config:
    config = yaml.safe_load(config)


def preprocess(name):
    try:
        x, sr = li.load(name, sr=config["data"]["sampling_rate"])
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print(e)
        return None
    N = config["data"]["signal_size"]
    pad = (N - (len(x) % N)) % N
    x = np.pad(x, (0, pad))

    step_size = config["data"]["block_size"] / config["data"]["sampling_rate"]
    f0 = crepe.predict(x, sr, step_size=1000 * step_size)[1]

    loudness = li.feature.rms(x, hop_length=config["data"]["block_size"])
    loudness = np.log(loudness + 1e-5)

    x = x.reshape(-1, N)
    crop = N // config["data"]["block_size"] * x.shape[0]
    f0 = f0[..., :crop].reshape(x.shape[0], -1)
    loudness = loudness[..., :crop].reshape(x.shape[0], -1)

    x = x.astype(np.float32)
    f0 = f0.astype(np.float32)
    loudness = loudness.astype(np.float32)

    return zip(x, f0, loudness)


trainloader = torch.utils.data.DataLoader(
    SimpleDataset(config["data"]["preprocessed"],
                  config["data"]["wav_loc"],
                  preprocess_function=preprocess,
                  split_set="full"),
    config["training"]["batch"],
    shuffle=True,
    drop_last=True,
)

mean_loudness = 0
std_loudness = 0
iteration = 0
for x, f0, loudness in tqdm(trainloader):
    iteration += 1
    mean_loudness += (loudness.mean() - mean_loudness) / iteration
    std_loudness += (loudness.std() - std_loudness) / iteration

config["postproc"] = {}
config["postproc"]["mean_loudness"] = mean_loudness.item()
config["postproc"]["std_loudness"] = std_loudness.item()

root = path.join(config["training"]["root"], config["training"]["name"])
writer = SummaryWriter(root, flush_secs=20)

with open(
        path.join(
            config["training"]["root"],
            config["training"]["name"],
            "config.yaml",
        ), "w") as out:
    yaml.safe_dump(config, out)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DDSP(
    config["recurrent_args"],
    config["harmonic_args"],
    config["noise_args"],
    config["training"]["scales"],
).to(device)
opt = torch.optim.Adam(model.parameters(), config["training"]["lr"])

for e in range(config["training"]["epochs"]):
    for x, f0, loudness in tqdm(trainloader):
        x = x.to(device)
        f0 = f0.unsqueeze(-1).to(device)
        loudness = loudness.unsqueeze(-1).to(device)

        y = model(f0, loudness).squeeze(1)

        with torch.no_grad():
            Sx = model.multiScaleFFT(x)

        Sy = model.multiScaleFFT(y)

        loss = sum([(sx - sy).abs().mean() for sx, sy in zip(Sx, Sy)])
        loss.backward()
        opt.step()