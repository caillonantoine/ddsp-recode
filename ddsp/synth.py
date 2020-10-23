import torch
import torch.nn as nn
import math
from .complex_utils import complex_abs, complex_product


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x).pow(math.log(10)) + 1e-7


class Synth(nn.Module):
    @property
    def f0_cdt(self):
        return False

    @property
    def lo_cdt(self):
        return False


class Harmonic(Synth):
    @property
    def f0_cdt(self):
        return True

    def __init__(self, feature_out, n_harmonic, upsample_factor,
                 sampling_rate):
        super().__init__()
        self.n_harmonic = n_harmonic

        self.proj_alphas = nn.Linear(feature_out, n_harmonic)
        self.proj_amplitude = nn.Linear(feature_out, 1)
        self.upsample_factor = upsample_factor
        self.sampling_rate = sampling_rate

        self.upsample = nn.Upsample(
            scale_factor=upsample_factor,
            mode="linear",
            align_corners=True,
        )

    def forward(self, x, f0):
        f0 = f0.reshape(f0.shape[0], 1, -1)

        alphas = mod_sigmoid(self.proj_alphas(x))
        alphas = alphas / alphas.sum(-1, keepdim=True)
        alphas = self.upsample(alphas.transpose(1, 2))

        amplitude = mod_sigmoid(self.proj_amplitude(x))
        amplitude = self.upsample(amplitude.transpose(1, 2))

        with torch.no_grad():
            f0 = self.upsample(f0)

        h_index = torch.arange(1, self.n_harmonic + 1).reshape(1, -1, 1).to(x)
        phase = 2 * math.pi * torch.cumsum(f0, -1) / self.sampling_rate
        phase = phase % (2 * math.pi)
        phase = phase * h_index

        f0s = f0 * h_index
        antialiasing = f0s < self.sampling_rate / 2

        x = (torch.cos(phase) * antialiasing * alphas).sum(1, keepdim=True)
        x = x * amplitude

        return x


class Noise(Synth):
    def __init__(self, feature_out, n_band, upsample_factor, sampling_rate):
        super().__init__()

        self.feature_out = feature_out
        self.n_band = n_band
        self.upsample_factor = upsample_factor
        self.sampling_rate = sampling_rate

        self.proj_noise = nn.Linear(feature_out, n_band)

        window = torch.hann_window(n_band * 2 - 1)
        self.register_buffer("window", window)

    def window_filters(self, x):
        x = x.unsqueeze(-1).expand(*x.shape, 2).contiguous()
        x[..., 1] = 0
        x = torch.irfft(x, 1, normalized=True)
        x = torch.roll(x, self.n_band)
        x = x * self.window
        x = torch.nn.functional.pad(x, (0, self.upsample_factor - x.shape[-1]))
        x = torch.rfft(x, 1, normalized=True)
        return x

    def forward(self, x):
        x = mod_sigmoid(self.proj_noise(x))
        x = self.window_filters(x)
        # x = torch.view_as_complex(x)

        noise = torch.rand(x.shape[0], x.shape[1], self.upsample_factor)
        noise = 2 * noise - 1
        noise = torch.rfft(noise, 1, normalized=True).to(x)
        # noise = torch.view_as_complex(noise).to(x)

        noise = complex_product(x, noise)

        # noise = torch.view_as_real(noisse)
        noise = torch.irfft(noise, 1, normalized=True)[..., :-1]
        noise = noise.reshape(noise.shape[0], 1, -1)

        return noise


class Reverb(Synth):
    def __init__(self):
        super().__init__()
        self.wet = nn.Parameter(torch.tensor(.5))
        self.decay = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        t = torch.arange(x.shape[-1]).to(x)
        wet = (2 * torch.rand_like(x) - 1)
        wet = torch.exp(-torch.exp(self.decay) * t) * wet

        dry = torch.zeros_like(x)
        dry[..., 0] = 1

        impulse = wet * self.wet + dry * (1 - self.wet)
        impulse = torch.rfft(impulse, 1, normalized=True)

        x = torch.rfft(x, 1, normalized=True)
        x = complex_product(impulse, x)
        x = torch.irfft(x, 1, normalized=True)[..., :t.shape[-1]]

        return x