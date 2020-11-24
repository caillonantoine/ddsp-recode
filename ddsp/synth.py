import torch
import torch.nn as nn
import math
from .complex_utils import complex_abs, complex_product
import torch.fft as fft


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x).pow(math.log(10)) + 1e-7


class Harmonic(nn.Module):
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
        alphas = mod_sigmoid(self.proj_alphas(x))
        alphas = self.upsample(alphas.transpose(1, 2))
        alphas = alphas / alphas.sum(1, keepdim=True)

        amplitude = nn.functional.softplus(self.proj_amplitude(x))
        amplitude = self.upsample(amplitude.transpose(1, 2))

        f0 = f0.reshape(f0.shape[0], 1, -1)
        f0 = self.upsample(f0) / self.sampling_rate

        h_index = torch.arange(1, self.n_harmonic + 1).reshape(1, -1, 1).to(x)

        phase = 2 * math.pi * torch.cumsum(f0, -1)
        phase = phase * h_index

        f0s = f0 * h_index
        antialiasing = (f0s < .5).float()

        x = (torch.sin(phase) * alphas * antialiasing).sum(1, keepdim=True)
        x = x * amplitude

        return x, {"amp": amplitude, "alphas": alphas}


class Noise(nn.Module):
    def __init__(self, feature_out, n_band, upsample_factor, sampling_rate):
        super().__init__()

        self.feature_out = feature_out
        self.n_band = n_band
        self.upsample_factor = upsample_factor
        self.sampling_rate = sampling_rate

        self.proj_noise = nn.Linear(feature_out, n_band)

        window = torch.hann_window((n_band - 1) * 2)
        self.register_buffer("window", window)

    def window_filters(self, x):
        x = fft.irfft(x)
        x = torch.roll(x, self.n_band)
        x = x * self.window
        x = torch.nn.functional.pad(x, (0, self.upsample_factor - x.shape[-1]))
        x = fft.rfft(x)
        return x

    def forward(self, x):
        x = mod_sigmoid(self.proj_noise(x))
        x = self.window_filters(x)

        noise = torch.rand(x.shape[0], x.shape[1],
                           self.upsample_factor).to(x.device)
        noise = (2 * noise - 1) / 100
        noise = fft.rfft(noise)

        noise = x * noise

        noise = fft.irfft(noise)
        noise = noise.reshape(noise.shape[0], 1, -1)

        return noise


class Reverb(nn.Module):
    def __init__(self, sampling_rate):
        super().__init__()
        self.wet = nn.Parameter(torch.tensor(0.))
        self.decay = nn.Parameter(torch.tensor(4.))
        self.sampling_rate = sampling_rate

    def forward(self, x):
        noise = 2 * torch.rand_like(x) - 1
        t = torch.arange(noise.shape[-1]).to(x.device) / self.sampling_rate
        ramp = torch.exp(-torch.exp(self.decay) * t)
        noise = noise * ramp

        identity = torch.zeros_like(noise)
        identity[..., 0] = 1

        wet = torch.sigmoid(self.wet)
        impulse = identity * (1 - wet) + noise * wet

        impulse = impulse / impulse.sum(-1, keepdim=True)

        N = x.shape[-1]
        x = nn.functional.pad(x, (N, 0))
        impulse = nn.functional.pad(impulse, (0, N))

        impulse_f = fft.rfft(impulse)
        x = fft.rfft(x)

        x = impulse_f * x

        x = fft.irfft(x)[..., N:]

        return x, {"ramp": ramp, "noise": noise, "impulse": impulse}
