# Learning Controls and Interactions for DDSP

## Pretrained models

In this directory you'll find three subdirectories:

* `solordinario`
* `johngarner`
* `saxophone`

each containing a **pretrained** model, the **dataset** on which the model has been trained, and a **configuration** file.


### 3 - 4. VAE for fixed-length and conditional samples

The first one (`solordinario`) is a subset of the **Sol** library made by Ircam, composed of **single** notes from a violin. Each wav file is annotated with a **pitch** and an **amplitude** that you can use as **labels**. This dataset must be used as a base for sections 3 and 4. We graciously give you this dataset for this project but remember **that this is a paid dataset, hence use it only for this project and nothing else** (including sharing it, selling it...)

### 5. Recurrent approach for variable-length samples

The two other directories (`johngarner` and `saxophone`) should be used as a base for section 5, since there are composed of 15-30mn of single instrument performance. Those performances have been downloaded from youtube, hence you can use it for any non-commercial purpose.

## Use case

Say you want to analyse and reconstruct the following audio `solordinario/data/Vn-ord-ff-1c-_A#6.wav` using the **solordinario** pretrained model. You can use the following script:

```python
import torch
import librosa as li
import descriptors

audio, sampling_rate = li.load(
    'solordinario/data/Vn-ord-ff-1c-_A#6.wav',
    sr=16000,
)

pitch = descriptors.extract_pitch(
    audio,
    sampling_rate=sampling_rate,
    block_size=64,
)

loudness = descriptors.extract_loudness(
    audio,
    sampling_rate=sampling_rate,
    block_size=64,
)

pitch = torch.from_numpy(pitch).float().reshape(1, -1, 1)
loudness = torch.from_numpy(loudness).float().reshape(1, -1, 1)

model = torch.jit.load('solordinario/ddsp_solordinario_pretrained.ts')

reconstruction = model(pitch, loudness)
```

Both sampling rate and block size for a particular model can be found in its configuration file.