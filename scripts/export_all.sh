runs="runs/johngarner runs/saxophone runs/solordinario"

mkdir temp
cp scripts/readme.md scripts/example.py temp/

for run in $runs;
do
    echo exporting run $(basename $run)
    python export.py --out-dir temp/$(basename $run) --data true --run $run
done

echo creating archive
cd temp

echo "import numpy as np
import librosa as li
import crepe

def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sampling_rate, n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S


def extract_pitch(signal, sampling_rate, block_size):
    f0 = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=0,
        center=True,
        viterbi=True,
    )
    return f0[1].reshape(-1)[:-1]" > descriptors.py

tar -czf ../ddsp.tar.gz *
cd ../
rm -fr temp
