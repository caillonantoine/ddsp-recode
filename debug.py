#%%
import torch
import librosa as li
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt

x, sr = li.load("runs/test_violin_harmonic_hilr/audio_000000.wav", 16000)

#%% RMS SIMPLE
loudness = rearrange(
    x,
    "(block time) -> block time",
    time=160,
)**2
win = np.hamming(160)
win /= np.sum(win)
loudness = np.sum(loudness * win, -1)
loudness = np.log(loudness + 1e-4)
print(loudness.shape)

plt.plot(loudness)
plt.xlim([0, 2000])
plt.show()
# %% RMS SPEC
S = li.stft(x, n_fft=2048, hop_length=160, win_length=2048, center=True)
S = abs(S)
loudness = li.feature.rms(S=S, frame_length=2048,
                          center=True).reshape(-1)[..., :-1]
loudness = np.log(loudness + 1e-4)
print(loudness.shape)

plt.plot(loudness)
plt.xlim([0, 2000])
plt.show()
# %%
