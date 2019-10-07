import central_training as ct
from ddsp import NeuralSynth
from loader import Loader
import torch
import numpy as np

def train_step(model, opt_list, step, data_list):
    lo = data_list.pop(0)
    f0 = data_list.pop(0)
    stfts = data_list

    output = model(f0.unsqueeze(-1),
                   lo.unsqueeze(-1))

    stfts_rec = model.multiScaleFFT(output)

    print([stft.shape for stft in stfts_rec])
    print([stft.shape for stft in stfts])

trainer = ct.Trainer(**ct.args.__dict__)

trainer.set_model(NeuralSynth)
trainer.setup_model()

trainer.add_optimizer(torch.optim.Adam(trainer.model.parameters()))
trainer.setup_optim()

trainer.set_dataset_loader(Loader)
trainer.set_lr(np.linspace(1e-4, 2e-5, ct.args.step))

trainer.set_train_step(train_step)

for elm in trainer.train_loop():
    break
