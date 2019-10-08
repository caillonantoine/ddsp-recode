import numpy as np
import soundfile as sf
import librosa as li
from tqdm import tqdm
import os
from hparams import preprocess
from ddsp import NeuralSynth
import torch

multiScaleFFT = NeuralSynth().multiScaleFFT
amp = lambda x: x[:,:,0]**2 + x[:,:,1]**2

def process(filename, block_size, sequence_size):
    os.makedirs("output", exist_ok=True)

    sound = sf.SoundFile(filename)
    batch = len(sound) // (block_size * sequence_size)

    print(f"Splitting data into {batch} examples of {sequence_size}-deep sequences of {block_size} samples.")

    scales = preprocess.fft_scales
    output = preprocess.output_dir
    sp = []
    for scale, ex_sp in zip(scales,multiScaleFFT(torch.randn(block_size * sequence_size),amp=amp)):
        sp.append(np.memmap(f"{output}/sp_{scale}.npy",
                            dtype=np.float32,
                            shape=(batch, ex_sp.shape[0], ex_sp.shape[1]),
                            mode="w+"))

    lo = np.zeros([batch,sequence_size])

    for b in tqdm(range(batch)):
        x = sound.read(block_size * sequence_size)
        for i,msstft in enumerate(multiScaleFFT(torch.from_numpy(x).float(), amp=amp)):
            sp[i][b,:,:] = msstft.detach().numpy()

        x = x.reshape(-1, block_size)
        for i,seq in enumerate(x):
            lo[b,i] = np.log(np.mean(seq**2)+1e-15)

    mean_loudness = np.mean(lo)
    std_loudness  = np.std(lo)
    lo -= mean_loudness
    lo /= std_loudness

    np.save(f"{output}/lo.npy", lo)


if __name__ == '__main__':
    process(preprocess.input_filename,
            preprocess.block_size,
            preprocess.sequence_size)
