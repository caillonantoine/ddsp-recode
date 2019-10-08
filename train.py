import central_training as ct
from ddsp import NeuralSynth
from loader import Loader
import torch
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import os
from torch.utils.tensorboard import SummaryWriter

os.makedirs("temp", exist_ok=True)



def train_step(model, opt_list, step, data_list):
    opt_list[0].zero_grad()
    lo = data_list.pop(0)
    f0 = data_list.pop(0)
    stfts = data_list

    output, amp, alpha, S_noise = model(f0.unsqueeze(-1),
                                  lo.unsqueeze(-1))
                                  
    # torch.Size([8, 25600]) (output)
    # torch.Size([8, 25600]) (amp)
    # torch.Size([8, 25600, 10]) (alpha)
    # torch.Size([8, 400, 33, 2]) (S_noise)


    stfts_rec = model.multiScaleFFT(output)

    lin_loss = sum([torch.mean(abs(stfts[i]**2 - stfts_rec[i])) for i in range(len(stfts_rec))])
    log_loss = sum([torch.mean(torch.log(abs(stfts[i]**2 - stfts_rec[i]) + 1e-4)) for i in range(len(stfts_rec))])


    loss = 10 * lin_loss + log_loss
    loss.backward()
    opt_list[0].step()

    if step % 10 == 0:
        plt.figure(figsize=(15,5))
        plt.subplot(131)
        plt.plot(output[0].detach().cpu().numpy().reshape(-1))
        plt.title("Rec waveform")

        plt.subplot(132)
        plt.imshow(np.log(stfts[2][0].cpu().detach().numpy()+1e-3), cmap="magma", origin="lower", aspect="auto")
        plt.title("Original spectrogram")
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(np.log(stfts_rec[2][0].cpu().detach().numpy()+1e-3), cmap="magma", origin="lower", aspect="auto")
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
trainer.set_lr(np.linspace(1e-4, 2e-5, ct.args.step))

trainer.set_train_step(train_step)

writer = SummaryWriter(f"runs/{ct.args.name}/")

for i,losses in enumerate(trainer.train_loop()):
    for loss in losses:
        writer.add_scalar(loss, losses[loss], i)
