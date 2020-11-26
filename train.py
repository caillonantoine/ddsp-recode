from os import path
from types import SimpleNamespace

import torch
import numpy as np
import librosa as li
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

import crepe
from udls import SimpleDataset
from torch.utils.tensorboard import SummaryWriter

from ddsp.model import DDSP

import soundfile as sf
from einops import rearrange

import logging
logging.basicConfig(level=logging.ERROR)

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

    S = li.stft(x,
                n_fft=512,
                hop_length=config["data"]["block_size"],
                win_length=512,
                center=True)
    S = abs(S)
    loudness = li.feature.rms(
        S=S,
        frame_length=512,
        center=True,
    ).reshape(-1)[..., :-1]
    loudness = np.log(loudness + 1e-4)

    x = x.reshape(-1, N)
    loudness = loudness.reshape(x.shape[0], -1)

    f0 = []
    crop = N // config["data"]["block_size"]
    for sample in x:
        f0.append(
            crepe.predict(
                np.pad(sample, (0, 10 * config["data"]["block_size"])),
                sr,
                step_size=1000 * step_size,
                verbose=0,
                center=True,
                viterbi=True,
            )[1][:crop])
    f0 = np.stack(f0, 0)
    loudness = loudness[..., :crop * x.shape[0]].reshape(x.shape[0], -1)

    x = x.astype(np.float32)
    f0 = f0.astype(np.float32)
    loudness = loudness.astype(np.float32)

    return zip(x, f0, loudness)


trainloader = torch.utils.data.DataLoader(
    SimpleDataset(config["data"]["preprocessed"],
                  config["data"]["wav_loc"],
                  preprocess_function=preprocess,
                  split_set="full",
                  extension="*.wav,*.mp3"),
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
    config["scales"],
).to(device)
opt = torch.optim.Adam(model.parameters(), config["training"]["lr"])

step = 0
for e in range(config["training"]["epochs"]):
    for x, f0, loudness in tqdm(trainloader):
        logging.debug("sending data to device")

        x = x.to(device)

        f0 = f0.unsqueeze(1).to(device)
        loudness = loudness.unsqueeze(1).to(device)

        loudness = (loudness - mean_loudness) / std_loudness

        logging.debug("forward pass")
        y, artifacts = model(f0, loudness)
        y = y.squeeze(1)

        logging.debug("compute original multiscale")
        Sx = model.multiScaleStft(x)

        logging.debug("compute synthed multiscale")
        Sy = model.multiScaleStft(y)

        logging.debug("compute loss")
        lin_loss = 0
        log_loss = 0
        for sx, sy in zip(Sx, Sy):
            lin_loss = lin_loss + (sx - sy).abs().mean()
            log_loss = log_loss + (torch.log(sx + 1e-6) -
                                   torch.log(sy + 1e-6)).abs().mean()

        loss = lin_loss + log_loss
        if step < 1000:
            loss -= .1 * torch.log(artifacts["amp"].mean())

        logging.debug("backward pass")
        loss.backward()

        logging.debug("step")
        opt.step()

        if not step % 1000:
            plt.plot(artifacts["impulse"].cpu().detach().reshape(-1))
            plt.tight_layout()
            writer.add_figure("impulse", plt.gcf(), step)

            for scale, sx, sy in zip(config["scales"], Sx, Sy):
                plt.subplot(121)
                plt.imshow(sx[0].cpu().detach().numpy(),
                           aspect="auto",
                           origin="lower")
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(sy[0].cpu().detach().numpy(),
                           aspect="auto",
                           origin="lower")
                plt.colorbar()
                plt.tight_layout()
                writer.add_figure(f"scale {scale}", plt.gcf(), step)

            alpha_n = artifacts["alphas"].cpu().detach().numpy()
            histogram = [
                np.histogram(
                    alpha_n[:, i, :].reshape(-1),
                    bins=100,
                    range=(0, 1),
                )[0] for i in range(alpha_n.shape[1])
            ]
            histogram = np.asarray(histogram)
            plt.imshow(np.log(histogram.T + 1e-3),
                       origin="lower",
                       aspect="auto",
                       cmap="magma")
            plt.colorbar()
            plt.xlabel("Harmonic number")
            plt.ylabel("Density")
            plt.tight_layout()

            writer.add_figure(
                "harmonic_repartition",
                plt.gcf(),
                step,
            )

            plt.plot(artifacts["amp"][0].reshape(-1).cpu().detach().numpy())
            plt.xlabel("Time step")
            plt.ylabel("Amplitude")
            plt.tight_layout()

            writer.add_figure(
                "amplitude",
                plt.gcf(),
                step,
            )

            Sx = np.log(Sx[0][0].cpu().detach().numpy() + 1e-3)
            Sy = np.log(Sy[0][0].cpu().detach().numpy() + 1e-3)

            S = np.stack([Sx, Sy], 0)
            S = rearrange(S, "b w h -> (b w) h")

            plt.imshow(S, aspect="auto", origin="lower")
            plt.colorbar()
            plt.tight_layout()
            writer.add_figure("reconstruction", plt.gcf(), step)

            plt.plot(y[0].reshape(-1).cpu().detach().numpy())
            writer.add_figure("synth_signal", plt.gcf(), step)

            plt.plot(x[0].reshape(-1).cpu().detach().numpy())
            writer.add_figure("source_signal", plt.gcf(), step)

            plt.plot(f0[0].reshape(-1).cpu().detach().numpy())
            writer.add_figure("pitch", plt.gcf(), step)

            plt.plot(loudness[0].reshape(-1).cpu().detach().numpy())
            writer.add_figure("loudness", plt.gcf(), step)

            audio = torch.cat([x, y], -1).reshape(-1)
            sf.write(
                path.join(root, f"audio_{step:06d}.wav"),
                audio.cpu().detach().numpy(),
                model.harm_synth.sampling_rate,
            )

            torch.save(model.state_dict(), path.join(root, "state.pth"))

        writer.add_scalar("loss", loss.item(), step)
        step += 1
