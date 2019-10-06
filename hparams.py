class preprocess:
    input_filename = "flute_24.wav" # must be a mono, 44100Hz .wav file
    crepe_f0       = "output/flute.f0.csv"
    output_dir     = "output"

    fft_scales     = [2048, 1024, 512, 256, 128, 64] # Multi scale stft objective
    block_size     = 240 # Must be the same block size than that of crepe !!
    sequence_size  = 250 # Number of sequence to process in the GRU cell
