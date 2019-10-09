import torch
import torch.nn as nn
import numpy as np
from hparams import preprocess, ddsp

def mod_sigmoid(x):
    return 2*torch.sigmoid(x)**np.log(10) + 1e-7

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
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(ddsp.hidden_size,
                               ddsp.n_partial,
                               ddsp.filter_size)
        self.condition_upsample = nn.Upsample(scale_factor=preprocess.block_size,
                                              mode="linear")

        self.impulse = nn.Parameter(torch.zeros(1,
                                    preprocess.block_size * preprocess.sequence_size),
                                    requires_grad=False)

        for n,p in self.named_parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                # print(f"Skipped initialization of {n}")
                pass

        # self.impulse.data /= 10
        self.impulse.data[:,0] = 1
        self.impulse.data[0,:] *= torch.exp(-10*torch.linspace(0,1,
        preprocess.block_size * preprocess.sequence_size))



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


    def forward(self, f0, lo):
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



        # y =  lo * torch.sum(torch.sin(self.k * phi),-1)
        y =  amp * torch.sum(alpha * torch.sin( self.k * phi),-1)
        # y =  torch.sum(alpha * torch.sin( self.k * phi),-1)

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

        y += filtered_noise


        # CONVOLUTION WITH AN IMPULSE RESPONSE #################################
        Y_S = torch.rfft(y,1)

        if y.shape[-1] > preprocess.sequence_size * preprocess.block_size:
            self.impulse.data = nn.functional.pad(self.impulse.data,
                                                 (0, y.shape[-1]-self.impulse.shape[-1]),
                                                 "constant", 0)

        IR_S = torch.rfft(torch.tanh(self.impulse),1).expand_as(Y_S)
        Y_S_CONV = torch.zeros_like(IR_S)
        Y_S_CONV[:,:,0] = Y_S[:,:,0] * IR_S[:,:,0] - Y_S[:,:,1] * IR_S[:,:,1]
        Y_S_CONV[:,:,1] = Y_S[:,:,0] * IR_S[:,:,1] + Y_S[:,:,1] * IR_S[:,:,0]

        y_conv = torch.irfft(Y_S_CONV, 1, signal_sizes=(y.shape[-1],))

        return y_conv, amp, alpha, S_filtered_noise.reshape(bs,
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

    parser = ArgumentParser(description="Reconstruction of an input audio sample.")
    parser.add_argument("input",type=str, help="Audio to reconstruct")
    parser.add_argument("--state", type=str, default=None, help="Model state to load")
    args = parser.parse_args()

    x,fs = li.load(args.input, preprocess.samplerate)
    step_size = 1000 * preprocess.block_size // preprocess.samplerate

    f0 = crepe.predict(x,preprocess.samplerate,step_size=step_size)[1]
    lo = li.feature.rms(x, frame_length=64, hop_length=64, center=False)
    lo = np.log(lo**2 + 1e-15)

    mean, std = np.mean(lo), np.std(lo)

    lo -= mean
    lo /= std

    N = min(f0.shape[-1],lo.shape[-1])
    f0, lo = f0[:N],lo[:,:N]

    f0 = torch.from_numpy(f0).float().reshape(1,-1,1)
    lo = torch.from_numpy(lo).float().reshape(1,-1,1)


    NS = NeuralSynth()
    if args.state is not None:
        state = torch.load(args.state)[1]
        NS.load_state_dict(state)
    # NS.cuda()

    out,_,_,_ = NS(f0,lo)
    out = out.detach().cpu().numpy().reshape(-1)

    sf.write("reconstruction.wav", out, preprocess.samplerate)


















#
