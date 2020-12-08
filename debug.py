#%%
import librosa as li
from ddsp.core import extract_loudness
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()
# %%

x = np.load("/fast-1/tmp/john/signals.npy")[0]
sr = 16000

x = x[:2 * sr]

lo = extract_loudness(x, 16000, 160, n_fft=2048)
center = lambda x: (x - np.mean(x)) / np.std(x)

plt.plot(np.linspace(0, 1, x.shape[-1]), center(x))
plt.plot(np.linspace(0, 1, lo.shape[-1]), center(lo))
plt.show()
print(lo.shape)
# %%
