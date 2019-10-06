# @Author: Antoine Caillon <caillona>
# @Date:   2019-04-17T10:16:33+02:00
# @Email:  caillonantoine@gmail.com
# @Last modified by:   caillona
# @Last modified time: 2019-04-25T11:26:54+02:00

import torch
import json
import argparse
from torch.utils import data
from tqdm import tqdm
import torch.nn as nn


class Trainer:
    def __init__(self,
                 cuda,
                 step,
                 batch_size,
                 backup_every,
                 image_every,
                 resume,
                 name,
                 dataset,
                 hparams,
                 parallel,
                 **kwargs):
        if cuda == -1:
            self.device = torch.device("cpu")
        else:
            self.device       = torch.device(f"cuda:{cuda}")
        print("Using device %s" % str(self.device))

        self.step         = step
        self.batch_size   = batch_size
        self.backup_every = backup_every
        self.image_every  = image_every
        self.resume       = resume
        self.name         = name
        self.dataset      = dataset
        self.parallel     = parallel

        if hparams is not None:
            with open(hparams,"r") as inp:
                self.hparams = json.load(inp)
        else:
            self.hparams = None

        self.optim        = []

    def set_model(self, model):
        self.model = model

    def set_lr(self, lr):
        self.lr = lr

    def add_optimizer(self, optim):
        self.optim.append(optim)

    def set_dataset_loader(self, dataset_class):
        multiplier = max(1,self.parallel)
        self.SD       = dataset_class(self.dataset)
        self.SDloader = data.DataLoader(self.SD,
                                        batch_size=multiplier*self.batch_size,
                                        shuffle=True,
                                        drop_last=True)

    def set_train_step(self, train_step_function):
        self.train_step = train_step_function

    def setup_model(self):
        print(self.parallel)
        if self.resume is not None:
            state = torch.load(self.resume, map_location="cpu")

            if self.hparams is not None:
                self.model = self.model(**state[0])
            else:json
                self.model = self.model()


            if not self.parallel:
                self.model = self.model.to(self.device)
            self.model.load_state_dict(state[1])
            self.current_step = state[2] + 1
            print("Checkpoint resumed")


        else:
            print("No checkpoint to be resumed")
            if self.hparams is not None:
                self.model = self.model(**self.hparams)
            else:
                self.model = self.model()
            if not self.parallel:
                self.model = self.model.to(self.device)
            self.current_step = 0
json
        if self.parallel:
            self.model = nn.DataParallel(self.model, range(self.parallel))
            print("Serving in parallel...")
        self.model.train()

    def setup_optim(self):
        if self.resume is not None:
            state = torch.load(self.resume, map_location="cpu")

            for i,opt in enumerate(self.optim):
                opt.load_state_dict(state[i+3])

    def train_loop(self):
        current_step = self.current_step
        while current_step < self.step:
            for idx, data in enumerate(tqdm(self.SDloader, desc="train")):
                # torch.cuda.empty_cache()
                # UPDATE LEARNING RATE #########################################
                for opt in self.optim:
                    for p in opt.param_groups:
                        p["lr"] = self.lr[current_step]

                # UPLOAD DATA TO DEVICE ########################################
                for i in range(len(data)):
                    data[i] = data[i].to(self.device)

                # TRAIN STEP ###################################################
                yield self.train_step(self.model,
                                      self.optim,
                                      current_step,
                                      data)
                current_step += 1
                # BACKUP PASS ##################################################
                if current_step % self.backup_every == 0:
                    to_be_saved = [
                            self.hparams,
                            self.model.state_dict(),
                            current_step
                    ]
                    for opt in self.optim:
                        to_be_saved.append(opt.state_dict())
                    torch.save(to_be_saved,
                               f"runs/{self.name}/step_{current_step}.pth")




parser = argparse.ArgumentParser(description="Generic training script")
parser.add_argument("--cuda", type=int, default=-1, help="GPU id to use")
parser.add_argument("--parallel", type=int, default=0, help="Number of GPUs to use")
parser.add_argument("--step", type=int, default=100, help="Number of epoch to train on")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--backup-every", type=int, default=100, help="Do a backup every...")
parser.add_argument("--image-every", type=int, default=100, help="Do a image every...")
parser.add_argument("--resume", type=str, default=None, help="Pretrained model to resume")
parser.add_argument("--name", type=str, default="untitled", help="name of the session")
parser.add_argument("--hparams", type=str, default=None, help="hyper parameters to use")
parser.add_argument("dataset", type=str, help="Folder containing the dataset")
args = parser.parse_args()

#########
# USAGE #
#########
#
# import numpy as np
#
# trainer = Trainer(**args.__dict__)
#
# trainer.set_model(CycleGAN)
# trainer.setup_model()
#
# gen_opt = torch.optim.Adam([
#     {"params": trainer.model.GXY.parameters()},
#     {"params": trainer.model.GYX.parameters()}
# ])
#
# dis_opt = torch.optim.Adam([
#     {"params": trainer.model.DX.parameters()},
#     {"params": trainer.model.DY.parameters()}
# ])
#
# trainer.add_optimizer(gen_opt)
# trainer.add_optimizer(dis_opt)
# trainer.setup_optim()
#
# trainer.set_dataset_loader(Loader)
#
# trainer.set_lr(np.linspace(1e-3,1e-4, args.step))
