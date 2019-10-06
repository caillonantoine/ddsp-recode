import numpy as np
import soundfile as sf

def process(filename, block_size, sequence_size, fmin, fmax):
    for block in sf.blocks(filename, blocksize=block_size):
        
