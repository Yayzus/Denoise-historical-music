import pickle
import tqdm as tqdm
import numpy as np
from utils import convert_audio_to_image
import os

def read_list_from_file(filename):
    result = []
    with open(filename, "rb") as f:
        while True:
            try:
                chunk = pickle.load(f)
                result.extend(chunk)
            except EOFError:
                break
    return result

def main():

    if not os.path.exists('data/training_data'):
        os.makedirs('data/training_data')

    full_data = read_list_from_file("data/training_data.pickle")

    #Turning the waveform audio generated in noise_audio.py into 2 channel images for training
    for i, [clear, noisy] in enumerate(full_data):
        np.save(f'data/training_data/training_sample_{i}', np.stack((convert_audio_to_image(clear), convert_audio_to_image(noisy)), axis=0))

if __name__ == "__main__":
    main()