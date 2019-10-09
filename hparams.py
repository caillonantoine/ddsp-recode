class preprocess:
    input_filename = "data/cello.wav" # must be a mono, 16000Hz .wav file
    samplerate     = 16000 # Used when synth back audio
    output_dir     = "output"
    crepe_f0       = f"{output_dir}/cello.f0.csv"

    # Multi scale stft objective
    fft_scales     = [2048, 1024, 512, 256, 128, 64]

    # Must be the same block size than that of crepe !!
    block_size     = 64

    # Number of sequence to process in the GRU cell
    sequence_size  = 400

    # Must match the number displayed when preprocessing:
    # "Splitting data into XXXXX examples..."
    num_batch      = 1125

class ddsp:
    # Number of partials involved in the harmonic signal
    n_partial      = 100

    hidden_size    = 512
    filter_size    = 64
    impulse_time   = 2 # In second
