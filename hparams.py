class preprocess:
    input_filename = "flute.wav" # must be a mono, 44100Hz .wav file
    fft_scales     = [2048, 1024, 512, 256, 128, 64] # Multi scale stft objective
    output_dir     = "output"
    block_size     = 1024
    sequence_size  = 250
    fmin           = 250 # Used for pitch estimation. Could be replaced with CREPE
    fmax           = 2100
