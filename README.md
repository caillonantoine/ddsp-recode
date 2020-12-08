# Differentiable Digital Signal Processing

Implementation of the [DDSP model](https://github.com/magenta/ddsp) using PyTorch. This implementation can be exported to a torchscript model, ready to be used inside a realtime environment [WIP].

## Usage

Edit the `config.yaml` file to fit your needs (audio location, preprocess folder, sampling rate, model parameters...), then preprocess your data using 

```bash
python preprocess.py
```

You can then train your model using 

```bash
python train.py --name mytraining --epochs 10000000 --batch 16 --lr .001
```

Once trained, export it using

```bash
python export.py --run runs/mytraining/
```

It will produce a file named `ddsp_pretrained_mytraining.ts`, that you can use inside a python environment like that

```python
import torch

model = torch.jit.load("ddsp_pretrained_mytraining.ts")

pitch = torch.randn(1, 200, 1)
loudness = torch.randn(1, 200, 1)

audio = model(pitch, loudness)
```

## Realtime usage

A previous version of this repo allowed the use of the model as a **PureData external** (see [this video](https://www.youtube.com/watch?v=U2ZXANU9EQg)). Unfortunately it was using an unstable version of the **libtorch API** that is not usable anymore. A reimplementation of the realtime library will be available soon.