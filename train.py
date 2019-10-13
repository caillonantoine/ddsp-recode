import central_training as ct
from ddsp import NeuralSynth
from loader import Loader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_step(model, opt_list, step, data_list):
    noise_pass = True if step > 500 else False
    conv_pass  = True if step > 500 else False

    opt_list[0].zero_grad()
    lo = data_list.pop(0)
    f0 = data_list.pop(0)
    stfts = data_list

    output, amp, alpha, S_noise = model(f0.unsqueeze(-1),
                                  lo.unsqueeze(-1),
                                  noise_pass,
                                  conv_pass)

    stfts_rec = model.multiScaleFFT(output)

    lin_loss = sum([torch.mean(abs(stfts[i] - stfts_rec[i])) for i in range(len(stfts_rec))])
    log_loss = sum([torch.mean(abs(torch.log(stfts[i]+1e-4) - torch.log(stfts_rec[i] + 1e-4))) for i in range(len(stfts_rec))])


    loss = lin_loss + log_loss
    loss.backward()
    opt_list[0].step()

    if step % 50 == 0:
        # INFERED PARAMETERS PLOT ##############################################

        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.plot(np.log(amp[0].detach().cpu().numpy()))
        plt.title("Infered instrument amplitude")
        plt.xlabel("Time (ua)")
        plt.ylabel("Log amplitude (dB)")

        plt.subplot(132)

        alpha_n = alpha.cpu().detach().numpy()[0]
        histogram = [np.histogram(alpha_n[:,i], bins=100, range=(0,1))[0] for i in range(alpha_n.shape[-1])]
        histogram = np.asarray(histogram)
        plt.imshow(np.log(histogram.T+1e-3), origin="lower", aspect="auto", cmap="magma")
        plt.xlabel("Harmonic number")
        plt.ylabel("Density")
        plt.title("Harmonic repartition")

        plt.subplot(133)

        S_noise = S_noise[0].cpu().detach().numpy()
        S_noise = S_noise[:,:,0] ** 2 + S_noise[:,:,1] ** 2
        plt.imshow(S_noise.T, origin="lower", aspect="auto", cmap="magma")
        plt.title("Noise output")
        plt.xlabel("Time (ua)")
        plt.ylabel("Frequency (ua)")

        writer.add_figure("Infered parameters", plt.gcf(), step)
        plt.close()

        # RECONSTRUCTION PLOT ##################################################
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.plot(output[0].detach().cpu().numpy().reshape(-1))
        plt.title("Rec waveform")

        plt.subplot(132)
        plt.imshow(np.log(stfts[2][0].cpu().detach().numpy()+1e-4), cmap="magma", origin="lower", aspect="auto")
        plt.title("Original spectrogram")
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(np.log(stfts_rec[2][0].cpu().detach().numpy()+1e-4), cmap="magma", origin="lower", aspect="auto")
        plt.title("Reconstructed spectrogram")
        plt.colorbar()
        writer.add_figure("reconstruction", plt.gcf(), step)
        plt.close()

        writer.add_audio("Reconstruction", output[0].reshape(-1)/torch.max(output[0].reshape(-1)), step, 16000)

    return {"lin_loss":lin_loss.item(),
            "log_loss":log_loss.item()}

trainer = ct.Trainer(**ct.args.__dict__)

trainer.set_model(NeuralSynth)
trainer.setup_model()

trainer.add_optimizer(torch.optim.Adam(trainer.model.parameters()))
trainer.setup_optim()

trainer.set_dataset_loader(Loader)
trainer.set_lr(np.linspace(1e-3, 1e-4, ct.args.step))

trainer.set_train_step(train_step)

writer = SummaryWriter(f"runs/{ct.args.name}/")

for i,losses in enumerate(trainer.train_loop()):
    for loss in losses:
        writer.add_scalar(loss, losses[loss], i)
