import torch
import torch.nn as nn

from .base_layers import RecurrentBlock
from .synth import Harmonic, Noise, Reverb
from .complex_utils import complex_abs

import torch.fft as fft


class DDSP(nn.Module):
    def __init__(self, recurrent_args, harmonic_args, noise_args, scales):
        super().__init__()
        self.recurrent_block = RecurrentBlock(**recurrent_args)
        self.harmonic = Harmonic(**harmonic_args)
        self.noise = Noise(**noise_args)
        self.reverb = Reverb(harmonic_args["sampling_rate"])
        self.scales = scales

    def forward(self, pitch, loudness):
        hidden = self.recurrent_block(pitch, loudness)

        artifacts = {}

        y, _art = self.harmonic(hidden.clone(), pitch)
        artifacts.update(_art)

        noise = self.noise(hidden.clone())
        # y = y + noise

        y, _art = self.reverb(y)
        artifacts.update(_art)

        return y, artifacts

    def multiScaleFFT(self, x):
        stfts = []
        for scale in self.scales:
            S = torch.stft(
                x,
                n_fft=scale,
                window=torch.hann_window(scale).to(x),
                hop_length=int((.25) * scale),
                center=False,
                return_complex=True,
            )
            stfts.append(S.abs())
        return stfts