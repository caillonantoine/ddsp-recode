import torch
import torch.nn as nn

from .base_layers import RecurrentBlock
from .synth import Harmonic, Noise, Reverb


class DDSP(nn.Module):
    def __init__(self, recurrent_args, harmonic_args, noise_args):
        super().__init__()
        self.recurrent_block = RecurrentBlock(**recurrent_args)
        self.harmonic = Harmonic(**harmonic_args)
        self.noise = Noise(**noise_args)
        self.reverb = Reverb()

    def forward(self, pitch, loudness):
        hidden = self.recurrent_block([pitch, loudness])

        y = self.harmonic(hidden, pitch) + self.noise(hidden)
        y = self.reverb(y)

        return y
