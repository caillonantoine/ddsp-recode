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