import torch
import torch.nn as nn
import numpy as np
from hparams import preprocess

class MLP(nn.Module):
    def __init__(self, in_size=512, out_size=512, loop=3):
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            )] + [nn.Sequential(
                nn.Linear(out_size, out_size),
                nn.modules.normalization.LayerNorm(out_size),
                nn.ReLU()
            ) for i in range(loop - 1)]
        )

    def forward(self, x):
        for lin in self.linear:
            x = lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size, n_partial):
        super().__init__()
        self.f0_MLP = MLP(1,hidden_size)
        self.lo_MLP = MLP(1,hidden_size)

        self.gru    = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)

        self.fi_MLP = MLP(hidden_size, hidden_size)
        self.dense  = nn.Linear(hidden_size, n_partial)

    def forward(self, f0, lo):
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)

        x,_ = self.gru(torch.cat([f0, lo], -1))

        x = self.fi_MLP(x)

        alpha = torch.sigmoid(self.dense(x))
        print(alpha.shape)

        return alpha / torch.sum(alpha,-1).unsqueeze(-1)

class NeuralSynth(nn.Module):
    def __init__(self, n_partial):
        super().__init__()
        self.decoder = Decoder(512, n_partial)
        self.condition_upsample = nn.Upsample(scale_factor=preprocess.block_size,
                                              mode="linear")
        self.k = torch.arange(1, n_partial + 1)\
                      .reshape(1,1,-1)\
                      .float()\
                      .to(self.decoder.dense.weight.device)

        self.windows = [
            torch.from_numpy(
                np.hanning(scale)
            ).float().to(self.decoder.dense.weight.device)\
            for scale in preprocess.fft_scales
        ]


    def forward(self, f0, lo):
        assert len(f0.shape)==3, f"f0 input must be 3-dimensional, but is {len(f0.shape)}-dimensional."
        assert len(lo.shape)==3, f"lo input must be 3-dimensional, but is {len(lo.shape)}-dimensional."

        alpha = self.decoder(f0, lo)

        f0 = self.condition_upsample(f0.transpose(1,2)).squeeze(1)/preprocess.samplerate
        lo = self.condition_upsample(lo.transpose(1,2)).squeeze(1)
        alpha = self.condition_upsample(alpha.transpose(1,2)).transpose(1,2)

        phi = torch.zeros(f0.shape).to(f0.device)

        for i in np.arange(1,phi.shape[-1]):
            phi[:,i] = 2 * np.pi * f0[:,i] + phi[:,i-1]

        phi = phi.unsqueeze(-1).expand(alpha.shape)

        y =  lo * torch.sum(alpha * torch.sin( self.k * phi),-1)

        return y

    def multiScaleFFT(self, x, overlap=75/100):
        stfts = []
        amp = lambda x: torch.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2)
        for i,scale in enumerate(preprocess.fft_scales):
            stfts.append(amp(
                torch.stft(x,
                           n_fft=scale,
                           window=self.windows[i],
                           hop_length=int((1-overlap)*scale),
                           center=False)
            ))
        return stfts



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import librosa as li
    import sounddevice as sd
    plt.ion()

    ns = NeuralSynth(50)
    f0 = torch.from_numpy(np.linspace(110,220,preprocess.sequence_size).reshape(1,-1,1)).float()
    lo = torch.from_numpy(np.linspace(1,0,preprocess.sequence_size).reshape(1,-1,1)).float()
    output = ns(f0,lo)


























#
