import torch
import torch.nn as nn

from .base_layers import RecurrentBlock
from .synth import Harmonic, Noise, Reverb


class DDSP(nn.Module):
    def __init__(self, recurrent_args, harmonic_args, noise_args, scales):
        super().__init__()
        self.recurrent_block = RecurrentBlock(**recurrent_args)
        self.harmonic = Harmonic(**harmonic_args)
        self.noise = Noise(**noise_args)
        self.reverb = Reverb()
        self.scales = scales

    def forward(self, pitch, loudness):
        hidden = self.recurrent_block([pitch, loudness])

        y = self.harmonic(hidden, pitch)
        y += self.noise(hidden)
        y = self.reverb(y)

        return y

    def multiScaleFFT(self, x):
        stfts = []
        for scale in self.scales:
            S = torch.stft(
                x,
                n_fft=scale,
                window=torch.hann_window(scale).to(x),
                hop_length=int((.25) * scale),
                center=False,
            )
            stfts.append(torch.view_as_complex(S).abs())
        return stfts