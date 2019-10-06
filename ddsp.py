import torch
import torch.nn as nn
import numpy as np


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
    def __init__(self, hidden_size, n_harmonic):
        super().__init__()
        self.f0_MLP = MLP(1,hidden_size)
        self.lo_MLP = MLP(1,hidden_size)

        self.gru    = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)

        self.fi_MLP = MLP(hidden_size, hidden_size)
        self.dense  = nn.Linear(hidden_size, n_harmonic)

    def forward(self, f0, lo):
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)

        x,_ = self.gru(torch.cat([f0, lo], -1))

        x = self.fi_MLP(x)

        return self.dense(x)



class DDSP_AE(nn.Module):
    def __init__(self, n_harmonic, fmin, fmax):
        self.decoder = Decoder(512, n_harmonic)

if __name__ == '__main__':
    dec = Decoder(512, 100)
    f0  = torch.randn(1,250,1)
    lo  = torch.randn(1,250,1)
    dec(f0,lo)
