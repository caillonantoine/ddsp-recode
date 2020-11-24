import torch
import torch.nn as nn
import torch.fft as fft
import math


def upsampling(x, factor):
    return nn.functional.upsample(
        x,
        mode="linear",
        scale_factor=factor,
        align_corners=False,
    )


class HarmonicSynth(nn.Module):
    def __init__(self, sampling_rate, upsampling, n_harmonic):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsampling = upsampling
        self.n_harmonic = n_harmonic

    def forward(self, amplitude, alphas, pitch):
        amplitude = upsampling(amplitude, self.upsampling)
        alphas = upsampling(alphas, self.upsampling)
        pitch = upsampling(pitch, self.upsampling)
        pitch = pitch / self.sampling_rate

        indexes = torch.arange(1, self.n_harmonic + 1).reshape(1, -1, 1)
        indexes = indexes.to(alphas)

        phase = 2 * math.pi * torch.cumsum(pitch, -1)
        phase = indexes * phase

        aliasing = (pitch * indexes < .5).float()
        alphas = alphas * aliasing
        alphas /= alphas.sum(1, keepdim=True)

        y = (torch.sin(phase) * alphas * amplitude).sum(1, keepdim=True)

        return y


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

        bands = torch.roll(bands, bands.shape[-1] // 2, -1)
        bands *= self.window
        bands = nn.functional.pad(
            bands,
            (0, self.upsampling - 2 * (self.n_band - 1)),
        )
        bands = torch.roll(bands, bands.shape[-1] // 2, -1)

        noise = torch.rand_like(bands) * 2 - 1
        noise /= 100

        filt_noise = fft.irfft(fft.rfft(noise) * fft.rfft(bands))

        return filt_noise.reshape(filt_noise.shape[0], 1, -1)


class Reverb(nn.Module):
    def __init__(self, sampling_rate):
        super().__init__()
        self.wet = nn.Parameter(torch.tensor(0.))
        self.decay = nn.Parameter(torch.tensor(2.))
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

        return x