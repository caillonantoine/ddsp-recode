#%%
import torch
torch.set_grad_enabled(False)

model = torch.jit.load("full_model.ts")
f0 = torch.linspace(100, 200, 400).reshape(1, 1, -1)
lo = torch.randn(1, 1, 400)

y = model(f0, lo)

import librosa as li
import matplotlib.pyplot as plt
y = y.reshape(-1).cpu().numpy()

S = li.feature.melspectrogram(y)
S = li.amplitude_to_db(S)

plt.imshow(S)