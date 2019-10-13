import torch
import torch.nn as nn
import numpy as np
from hparams import preprocess, ddsp
import matplotlib.pyplot as plt


class Reverb(nn.Module):
    """
    Model of a room impulse response. Two parameters controls the shape of the
    impulse reponse: wet/dry amount and the exponential decay.

    Parameters
    ----------

    size: int
        Size of the impulse response (samples)
    """
    def __init__(self, size):
        super().__init__()
        self.size    = size

        self.impulse  = nn.Parameter(torch.rand(1, size) * 2 - 1, requires_grad=False)
        self.identity = nn.Parameter(torch.zeros(1,size), requires_grad=False)

        self.impulse[:,0]  = 0
        self.identity[:,0] = 1

        self.decay   = nn.Parameter(torch.Tensor([2]), requires_grad=True)
        self.wetdry  = nn.Parameter(torch.Tensor([4]), requires_grad=True)

    def forward(self):
        idx = torch.sigmoid(self.wetdry) * self.identity
        imp = torch.sigmoid(1 - self.wetdry) * self.impulse
        dcy = torch.exp(-(torch.exp(self.decay)+2) * \
              torch.linspace(0,1, self.size).to(self.decay.device))

        return idx + imp * dcy

def mod_sigmoid(x):
    """
    Implementation of the modified sigmoid described in the original article.

    Parameters
    ----------

    x: Tensor
        Input tensor, of any shape

    Returns
    -------

    Tensor:
        Output tensor, shape of x
    """
    return 2*torch.sigmoid(x)**np.log(10) + 1e-7

class MLP(nn.Module):
    """
    Implementation of a Multi Layer Perceptron, as described in the
    original article (see README)

    Parameters
    ----------

    in_size: int
        Input size of the MLP
    out_size: int
        Output size of the MLP
    loop: int
        Number of repetition of Linear-Norm-ReLU
    """
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
    """
    Decoder of the architecture described in the original architecture.

    Parameters
    ----------

    hidden_size: int
        Size of vectors inside every MLP + GRU + Dense
    n_partial: int
        Number of partial involved in the harmonic generation. (>1)
    filter_size: int
        Size of the filter used to shape noise.


    """
    def __init__(self, hidden_size, n_partial, filter_size):
        super().__init__()
        self.f0_MLP = MLP(1,hidden_size)
        self.lo_MLP = MLP(1,hidden_size)

        self.gru    = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)

        self.fi_MLP = MLP(hidden_size, hidden_size)

        self.dense_amp    = nn.Linear(hidden_size, 1)
        self.dense_alpha  = nn.Linear(hidden_size, n_partial)
        self.dense_filter = nn.Linear(hidden_size, filter_size // 2 + 1)

        self.n_partial = n_partial

    def forward(self, f0, lo):
        f0 = self.f0_MLP(f0)
        lo = self.lo_MLP(lo)

        x,_ = self.gru(torch.cat([f0, lo], -1))

        x = self.fi_MLP(x)

        amp          = mod_sigmoid(self.dense_amp(x))
        alpha        = mod_sigmoid(self.dense_alpha(x))
        filter_coeff = mod_sigmoid(self.dense_filter(x))

        alpha        = alpha / torch.sum(alpha,-1).unsqueeze(-1)

        return amp, alpha, filter_coeff

class NeuralSynth(nn.Module):
    """
    Implementation of a parametric Harmonic + Noise + IR reverb synthesizer,
    whose parameters are controlled by the previously implemented decoder.
    """
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(ddsp.hidden_size,
                               ddsp.n_partial,
                               ddsp.filter_size)

        self.condition_upsample = nn.Upsample(scale_factor=preprocess.block_size,
                                              mode="linear")

        for n,p in self.named_parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                # print(f"Skipped initialization of {n}")
                pass


        self.k = nn.Parameter(torch.arange(1, ddsp.n_partial + 1)\
                              .reshape(1,1,-1)\
                              .float(), requires_grad=False)

        self.windows = nn.ParameterList(
        nn.Parameter(torch.from_numpy(
                np.hanning(scale)
            ).float(), requires_grad=False)\
            for scale in preprocess.fft_scales)

        self.filter_window = nn.Parameter(torch.hann_window(ddsp.filter_size)\
                                               .roll(ddsp.filter_size//2,-1),
                                               requires_grad=False)

        self.impulse = Reverb(preprocess.sequence_size * preprocess.block_size)



    def forward(self, f0, lo, noise_pass=False, conv_pass=False):
        bs = f0.shape[0]
        assert len(f0.shape)==3, f"f0 input must be 3-dimensional, but is {len(f0.shape)}-dimensional."
        assert len(lo.shape)==3, f"lo input must be 3-dimensional, but is {len(lo.shape)}-dimensional."

        amp, alpha, filter_coef = self.decoder(f0, lo)


        f0          = self.condition_upsample(f0.transpose(1,2))\
                          .squeeze(1)/preprocess.samplerate

        amp         = self.condition_upsample(amp.transpose(1,2)).squeeze(1)
        alpha       = self.condition_upsample(alpha.transpose(1,2)).transpose(1,2)

        phi = torch.zeros(f0.shape).to(f0.device)

        for i in np.arange(1,phi.shape[-1]):
            phi[:,i] = 2 * np.pi * f0[:,i] + phi[:,i-1]

        phi = phi.unsqueeze(-1).expand(alpha.shape)

        antia_alias = (self.k * f0.unsqueeze(-1) < .5).float()

        y =  amp * torch.sum(antia_alias * alpha * torch.sin( self.k * phi),-1)

        # FREQUENCY SAMPLING FILTERING #########################################
        noise = torch.from_numpy(np.random.uniform(-1,1,y.shape))\
                     .float().to(y.device)/1000

        noise = noise.reshape(-1, ddsp.filter_size)
        S_noise = torch.rfft(noise,1).reshape(bs,-1,ddsp.filter_size//2+1,2)

        filter_coef = filter_coef.reshape([-1,
                                           ddsp.filter_size//2+1, 1])
        filter_coef = filter_coef.expand([-1,
                                          ddsp.filter_size//2+1, 2]).contiguous()

        filter_coef[:,:,1] = 0
        h = torch.irfft(filter_coef, 1, signal_sizes=(ddsp.filter_size,))
        h_w = self.filter_window.unsqueeze(0) * h

        H = torch.rfft(h_w, 1).reshape(bs, -1, ddsp.filter_size//2 + 1, 2)

        S_filtered_noise          = torch.zeros_like(H)
        S_filtered_noise[:,:,:,0] = H[:,:,:,0] * S_noise[:,:,:,0] - H[:,:,:,1] * S_noise[:,:,:,1]
        S_filtered_noise[:,:,:,1] = H[:,:,:,0] * S_noise[:,:,:,1] + H[:,:,:,1] * S_noise[:,:,:,0]

        S_filtered_noise          = S_filtered_noise.reshape(-1, ddsp.filter_size//2 + 1, 2)

        filtered_noise = torch.irfft(S_filtered_noise,1)[:,:ddsp.filter_size].reshape(bs,-1)

        if noise_pass:
            y += filtered_noise


        # CONVOLUTION WITH AN IMPULSE RESPONSE #################################
        y = nn.functional.pad(y, (0, preprocess.block_size*preprocess.sequence_size),
                              "constant", 0)

        Y_S = torch.rfft(y,1)

        impulse = self.impulse()

        impulse = nn.functional.pad(impulse,
                                    (0, preprocess.block_size*preprocess.sequence_size),
                                    "constant", 0)

        if y.shape[-1] > preprocess.sequence_size * preprocess.block_size:
            impulse = nn.functional.pad(impulse,
                                        (0, y.shape[-1]-impulse.shape[-1]),
                                        "constant", 0)
        if conv_pass:
            IR_S = torch.rfft(torch.tanh(impulse),1).expand_as(Y_S)
        else:
            IR_S = torch.rfft(torch.tanh(impulse.detach()),1).expand_as(Y_S)

        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[:,:,0] = Y_S[:,:,0] * IR_S[:,:,0] - Y_S[:,:,1] * IR_S[:,:,1]
        Y_S_CONV[:,:,1] = Y_S[:,:,0] * IR_S[:,:,1] + Y_S[:,:,1] * IR_S[:,:,0]

        y = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1],))

        y = y[:,:-preprocess.block_size*preprocess.sequence_size]

        return y, amp, alpha, S_filtered_noise.reshape(bs,
                                                       -1,
                                                       ddsp.filter_size//2+1, 2)

    def multiScaleFFT(self, x, overlap=75/100, amp = lambda x: x[:,:,:,0]**2 + x[:,:,:,1]**2):
        stfts = []
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
    import crepe
    import librosa as li
    from argparse import ArgumentParser
    import soundfile as sf
    from os import path


    parser = ArgumentParser(description="Reconstruction of an input audio sample.")
    parser.add_argument("input",type=str, help="Audio to reconstruct")
    parser.add_argument("--state", type=str, default=None, help="Model state to load")
    parser.add_argument("--transpose", type=int, default=0, help="Transposition amount (semitone)")
    parser.add_argument("--std-factor", type=float, default=1, help="Standard deviation modulation")
    args = parser.parse_args()

    x,fs = li.load(args.input, preprocess.samplerate)

    step_size = 1000 * preprocess.block_size // preprocess.samplerate

    if path.exists(path.splitext(args.input)[0] + "_f0.npy"):
        print("Found existing crepe analysis! Yay!")
        f0 = np.load(path.splitext(args.input)[0] + "_f0.npy")
    else:
        f0 = crepe.predict(x,preprocess.samplerate,step_size=step_size)[1]
        np.save(path.splitext(args.input)[0] + "_f0.npy", f0)


    lo = np.asarray([np.mean(x[i*preprocess.block_size:(i+1)*preprocess.block_size]**2)\
          for i in range(len(x)//preprocess.block_size)])
    lo = np.log(lo + 1e-15)
    lo = lo.reshape(1,-1)



    mean, std = np.mean(lo), np.std(lo)

    f0 *= 2**(args.transpose/12)
    std *= args.std_factor

    lo -= mean
    lo /= std

    N = min(f0.shape[-1],lo.shape[-1])
    f0, lo = f0[:N],lo[:,:N]

    f0 = torch.from_numpy(f0).float().reshape(1,-1,1)
    lo = torch.from_numpy(lo).float().reshape(1,-1,1)


    NS = NeuralSynth()
    if args.state is not None:
        state = torch.load(args.state, map_location="cpu")[1]
        NS.load_state_dict(state)

    with torch.no_grad():
        out,_,_,_ = NS(f0,lo, True, True)

    out = out.detach().cpu().numpy().reshape(-1)

    sf.write("reconstruction.wav", out, preprocess.samplerate)
