import torch
from torch.utils.tensorboard import SummaryWriter
from .synth import Harmonic, Noise, mod_sigmoid
from .model import DDSP

import matplotlib.pyplot as plt
import numpy as np
import librosa as li


class VizHook(object):
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.enabled = False
        self.step = 0

    @torch.no_grad()
    def __call__(self, module: torch.nn.Module, inputs, outputs):
        if self.enabled:
            if isinstance(module, Harmonic):
                self.harmonic_hook(module, inputs, outputs)
            elif isinstance(module, Noise):
                self.noise_hook(module, inputs, outputs)

    @torch.no_grad()
    def harmonic_hook(self, module: Harmonic, inputs, outputs):
        x, _ = inputs
        alphas = mod_sigmoid(module.proj_alphas(x))
        alphas = alphas / alphas.sum(-1, keepdim=True)  # bxtxc

        amplitude = mod_sigmoid(module.proj_amplitude(x)).squeeze().reshape(
            -1)  #bxtx1

        alpha_n = alphas.cpu().detach().numpy()
        histogram = [
            np.histogram(
                alpha_n[:, :, i].reshape(-1),
                bins=100,
                range=(0, 1),
            )[0] for i in range(alpha_n.shape[-1])
        ]
        histogram = np.asarray(histogram)
        plt.imshow(np.log(histogram.T + 1e-3),
                   origin="lower",
                   aspect="auto",
                   cmap="magma")
        plt.xlabel("Harmonic number")
        plt.ylabel("Density")
        plt.tight_layout()

        self.writer.add_figure(
            "harmonic_repartition",
            plt.gcf(),
            self.step,
        )

        plt.plot(amplitude.cpu().detach().numpy())
        plt.xlabel("Time step")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        self.writer.add_figure(
            "amplitude",
            plt.gcf(),
            self.step,
        )

    @torch.no_grad()
    def noise_hook(self, module: Noise, inputs, outputs):
        noise = outputs.cpu().numpy()[0].squeeze()
        noise_spec = li.feature.melspectrogram(noise, module.sampling_rate)
        noise_spec = np.log(noise_spec + 1e-3)

        plt.imshow(noise_spec)
        plt.tight_layout()

        self.writer.add_figure("noise", plt.gcf(), self.step)

    def enable_hook(self):
        self.enabled = True

    def disable_hook(self):
        self.enabled = False

    def set_step(self, value):
        self.step = value
