import pickle
import tqdm as tqdm
from torch.utils.data import random_split
import numpy as np
import librosa

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

def convert_audio_to_image(x):
    stft = librosa.stft(np.asarray(x), n_fft=1024, hop_length=512)
    real = np.real(stft)
    imaginary = np.imag(stft)

    img =  np.stack((real, imaginary), axis=0)

    return img

def main():
    full_data = read_list_from_file("data/training_data.pickle")

    for i, [clear, noisy] in enumerate(full_data):
        np.save(f'data/training_data/training_sample_{i}', np.stack((convert_audio_to_image(clear), convert_audio_to_image(noisy)), axis=0))

if __name__ == "__main__":
    main()