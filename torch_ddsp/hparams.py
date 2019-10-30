class preprocess:
    input_filename = "data/*.wav" # must be a mono, 16000Hz .wav file
    samplerate     = 16000 # Used when synth back audio
    output_dir     = "output"
    crepe_f0       = f"{output_dir}/violin_16.f0.csv"

    # Multi scale stft objective
    fft_scales     = [2048, 1024, 512, 256, 128, 64]

    # Block size used during feature extraction
    block_size     = samplerate // 100

    # Number of sequence to process in the GRU cell
    sequence_size  = 200

    # Must match the number displayed when preprocessing:
    # "Splitting data into XXXXX examples..."
    num_batch      = 898

    # Smoothed loudness kernel size
    kernel_size    = 8

class ddsp:
    # Encoder's convolutions stride
    strides          = [2,4,4,5]
    conv_hidden_size = 128
    conv_out_size    = 2
    conv_kernel_size = 15
    
    # Number of partials involved in the harmonic signal
    n_partial      = 100

    # Size of GRU hidden state
    hidden_size    = 512

    # Noise shaping filter size
    filter_size    = 64
