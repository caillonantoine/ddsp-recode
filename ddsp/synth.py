import torch
import torch.nn as nn
import torch.fft as fft
import math
from einops import rearrange


def upsampling(x, factor):
    return nn.functional.interpolate(
        x,
        mode="linear",
        scale_factor=factor,
        align_corners=False,
    )


class HarmonicSynth(nn.Module):
    def __init__(self, sampling_rate, upsampling, n_harmonic, smooth_size):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.register_buffer("upsampling", torch.tensor(upsampling))
        self.n_harmonic = n_harmonic

        win = torch.hamming_window(smooth_size)
        win /= torch.sum(win)
        self.register_buffer("smoothing_win", win.reshape(1, 1, -1))
        self.smooth_size = smooth_size

    def smooth_envelope(self, x):
        x = nn.functional.pad(
            x,
            (self.smooth_size // 2, self.smooth_size // 2),
            mode="reflect",
        )

        h = x.shape[1]

        # x = rearrange(x, "b h t -> (b h) t").unsqueeze(1)
        x = x.reshape(-1, 1, x.shape[-1])
        x = nn.functional.conv1d(x, self.smoothing_win)
        # x = rearrange(x.squeeze(1), "(b h) t -> b h t", h=h)
        x = x.reshape(-1, h, x.shape[-1])

        return x

    def forward(self, amplitude, alphas, pitch):
        amplitude = upsampling(amplitude, self.upsampling)
        amplitude = self.smooth_envelope(amplitude)

        alphas = upsampling(alphas, self.upsampling)
        alphas = self.smooth_envelope(alphas)

        pitch = upsampling(pitch, self.upsampling)
        pitch = pitch / self.sampling_rate

        indexes = torch.arange(1, self.n_harmonic + 1).reshape(1, -1, 1)
        indexes = indexes.to(alphas)

        phase = 2 * math.pi * torch.cumsum(pitch, -1)
        phase = indexes * phase

        aliasing = (pitch * indexes < .5).float()

        alphas = alphas * aliasing
        alphas /= alphas.sum(1, keepdim=True)

        y = amplitude * (torch.sin(phase) * alphas).sum(1, keepdim=True)

        return y, {"amp": amplitude, "alphas": alphas}


class NoiseSynth(nn.Module):
    def __init__(self, sampling_rate, n_band, upsampling):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_band = n_band
        self.upsampling = upsampling

        self.register_buffer(
            "window",
            torch.hann_window(2 * (self.n_band - 1)),
        )

    def forward(self, bands):
        # bands B N T
        bands = bands.transpose(1, 2)
        bands = fft.irfft(bands)
        bands = torch.roll(bands, self.n_band - 1, -1)
        bands = bands * self.window
        bands = nn.functional.pad(
            bands,
            (0, self.upsampling - 2 * (self.n_band - 1)),
        )
        bands = torch.roll(bands, -self.n_band + 1, -1)
        noise = torch.rand_like(bands) * 2 - 1

        filt_noise = fft.irfft(fft.rfft(noise) * fft.rfft(bands))

        return filt_noise.reshape(filt_noise.shape[0], 1, -1)


class Reverb(nn.Module):
    def __init__(self, sampling_rate):
        super().__init__()
        self.wet = nn.Parameter(torch.tensor(0.))
        self.decay = nn.Parameter(torch.tensor(2.))
        self.sampling_rate = sampling_rate
        self.impulse = nn.Parameter(
            torch.rand(1, 1, self.sampling_rate) * 2 - 1)

    def get_impulse(self, x):
        noise = nn.functional.pad(
            self.impulse,
            (0, x.shape[-1] - self.impulse.shape[-1]),
        )

        t = torch.arange(noise.shape[-1]).to(x.device) / self.sampling_rate
        ramp = torch.exp(-torch.exp(self.decay) * t)
        noise = noise * ramp

        identity = torch.zeros_like(noise)
        identity[..., 0] = 1

        wet = torch.sigmoid(self.wet)
        impulse = identity * (1 - wet) + noise * wet

        impulse = impulse / impulse.sum(-1, keepdim=True)
        return impulse

    def forward(self, x):
        impulse = self.get_impulse(x)

        N = x.shape[-1]
        x = nn.functional.pad(x, (N, 0))
        impulse = nn.functional.pad(impulse, (0, N))

        impulse_f = fft.rfft(impulse)
        x = fft.rfft(x)

        x = impulse_f * x

        x = fft.irfft(x)[..., N:]

        return x, {"impulse": impulse}
