import torch
import torch.nn as nn

from .base_layers import RecurrentBlock
from .synth import HarmonicSynth, NoiseSynth, Reverb

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

    def forward(self, pitch, loudness, noise_pass=True, reverb_pass=True):
        hidden = self.recurrent_block(
            pitch.permute(0, 2, 1),
            loudness.permute(0, 2, 1),
        )

        artifacts = {}

        amp = mod_sigmoid(self.amp_lin(hidden).permute(0, 2, 1))
        alphas = mod_sigmoid(self.alphas_lin(hidden).permute(0, 2, 1))

        harmonic, _art = self.harm_synth(amp, alphas, pitch)
        artifacts.update(_art)

        if noise_pass:
            bands = mod_sigmoid(self.bands_lin(hidden).permute(0, 2, 1) - 5)
            noise = self.noise_synth(bands)
            signal = harmonic + noise
        else:
            signal = harmonic

        if reverb_pass:
            mixdown, _art = self.reverb(signal)
            artifacts.update(_art)
        else:
            mixdown = signal

        return mixdown, artifacts

    def multiScaleStft(self, x):
        stfts = []
        for scale in self.scales:
            stfts.append(
                abs(
                    torch.stft(
                        x,
                        scale,
                        scale // 4,
                        scale,
                        center=True,
                        return_complex=True,
                        window=torch.hamming_window(scale).to(x),
                    )))
        return stfts


class ScriptableDDSP(DDSP):
    def forward(self, f0, lo):
        hidden = self.recurrent_block(
            f0.permute(0, 2, 1),
            lo.permute(0, 2, 1),
        )

        amp = mod_sigmoid(self.amp_lin(hidden).permute(0, 2, 1))
        alphas = mod_sigmoid(self.alphas_lin(hidden).permute(0, 2, 1))

        harmonic = self.harm_synth(amp, alphas, f0)[0]

        bands = mod_sigmoid(self.bands_lin(hidden).permute(0, 2, 1) - 5)
        noise = self.noise_synth(bands)
        signal = harmonic + noise

        if self.recurrent_block.cache.shape[0]:
            mixdown, _art = self.reverb(signal)
        else:
            mixdown = signal

        return mixdown
