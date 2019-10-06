import numpy as np
import soundfile as sf
import librosa as li
from tqdm import tqdm
import os
from hparams import preprocess

def multiScaleFFT(x, scales, overlap=75/100):
    stfts = []
    for scale in scales:
        stfts.append(abs(
            li.stft(x, n_fft=scale, hop_length=int((1-overlap)*scale), center=False)
        ))
    return stfts

def process(filename, block_size, sequence_size):
    os.makedirs("output", exist_ok=True)

    sound = sf.SoundFile(filename)
    batch = len(sound) // (block_size * sequence_size)

    print(f"Splitting data into {batch} examples of {sequence_size}-deep sequences of {block_size} samples.")

    scales = preprocess.fft_scales
    output = preprocess.output_dir
    sp = []
    for scale, ex_sp in zip(scales,multiScaleFFT(np.random.randn(block_size * sequence_size), scales)):
        sp.append(np.memmap(f"{output}/sp_{scale}.npy",
                            dtype=np.float32,
                            shape=(batch, ex_sp.shape[0], ex_sp.shape[1]),
                            mode="w+"))

    lo = np.zeros([batch,sequence_size])

    for b in tqdm(range(batch)):
        x = sound.read(block_size * sequence_size)
        for i,msstft in enumerate(multiScaleFFT(x, scales)):
            sp[i][b,:,:] = msstft

        x = x.reshape(-1, block_size)
        for i,seq in enumerate(x):
            lo[b,i] = np.sqrt(np.mean(seq**2))

    np.save(f"{output}/lo.npy", lo)


if __name__ == '__main__':
    process(preprocess.input_filename,
            preprocess.block_size,
            preprocess.sequence_size)
