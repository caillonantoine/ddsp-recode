from torch.utils.data import Dataset
import numpy as np
import torch
from hparams import preprocess

class Loader(Dataset):
    def __init__(self, dir):
        super().__init__()
        scales = preprocess.fft_scales
        # SPECTRAL FEATURES ####################################################
        self.sp = []
        for scale in scales:
            self.sp.append(
                np.memmap(f"{dir}/sp_{scale}.npy", dtype=np.float32)\
                .reshape(preprocess.num_batch, scale//2 + 1, -1)
            )

        # LOUDNESS #############################################################
        self.lo = np.load(f"{dir}/lo.npy")

        # CREPE F0 #############################################################
        N = self.lo.shape[0] * self.lo.shape[1]
        with open(preprocess.crepe_f0, "r") as crepe_f0:
            self.f0 = np.loadtxt(crepe_f0, delimiter=",", skiprows=1)[:N,1]
            self.f0 = self.f0.reshape([preprocess.num_batch, -1])

        self.scales = scales

    def __len__(self):
        return self.lo.shape[0]

    def __getitem__(self, i):
        memmap = [self.lo[i], self.f0[i]] + [sp[i] for sp in self.sp]
        return [torch.from_numpy(elm).float() for elm in memmap]


if __name__ == '__main__':
    loader = Loader("output")
