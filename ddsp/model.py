import torch
import torch.nn as nn

from .base_layers import RecurrentBlock
from .synth_bis import HarmonicSynth, NoiseSynth, Reverb
from .complex_utils import complex_abs

import torch.fft as fft
import math


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x).pow(math.log(10)) + 1e-7


class DDSP(nn.Module):
    def __init__(self, recurrent_args, harmonic_args, noise_args, scales):
        super().__init__()
        self.recurrent_block = RecurrentBlock(**recurrent_args)

        feature_out = recurrent_args["hidden_size"]

        self.amp_lin = nn.Linear(feature_out, 1)
        self.alphas_lin = nn.Linear(feature_out, harmonic_args["n_harmonic"])
        self.bands_lin = nn.Linear(feature_out, noise_args["n_band"])

        self.harm_synth = HarmonicSynth(**harmonic_args)
        self.noise_synth = NoiseSynth(**noise_args)

        self.reverb = Reverb(harmonic_args["sampling_rate"])

        self.scales = scales

    def forward(self, pitch, loudness):
        hidden = self.recurrent_block(
            pitch.permute(0, 2, 1),
            loudness.permute(0, 2, 1),
        )

        artifacts = {}

        amp = mod_sigmoid(self.amp_lin(hidden).permute(0, 2, 1))
        alphas = mod_sigmoid(self.alphas_lin(hidden).permute(0, 2, 1))
        bands = mod_sigmoid(self.bands_lin(hidden).permute(0, 2, 1))

        harmonic, _art = self.harm_synth(amp, alphas, pitch)
        artifacts.update(_art)

        noise = self.noise_synth(bands)

        mixdown, _art = self.reverb(harmonic + noise)
        artifacts.update(_art)

        return mixdown, artifacts

    def multiScaleStft(self, x):
        stfts = []
        for scale in self.scales:
            stfts.append(
                torch.stft(
                    x,
                    scale,
                    scale // 4,
                    scale,
                    center=True,
                    return_complex=True,
                    window=torch.hamming_window(scale).to(x),
                ).abs())
        return stfts