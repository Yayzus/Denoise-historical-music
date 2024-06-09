import numpy as np
import librosa
import pandas as pd
import pickle
import os
from tqdm import tqdm
from utils import normalize_audio


Q = 0.04 # 4%
T = 0.1  # window size in s
TARGET_SAMPLE_LENGTH = 0.25  # minimum length of desired noise samples is s

noise_samples = []

def extract_noise_samples(freq, sampleRate):
    window = int(sampleRate * T)
    freq = freq[int(0.4*sampleRate):-int(0.4*sampleRate)]
    
    #with calculating the rolling std we can have an idea about the loudness of the segment
    #Also for the first window_size elements this function doesnt calculate values (not enough data in the window).
    #With pandas.rolling.std our rolling_std variable would be constructed in a way where: the i-th element would be the std of the segment freq[i-window:i].
    #With removing the first window_size elements our rolling_std variable will contain the std for the segment freq[i:i+window]
    data = pd.Series(freq)
    rolling_std = data.rolling(window).std().tolist()[window-1:]

    treshold = np.quantile(rolling_std, Q)
    noise_sample_start = -1
    noise_sample_end = -1
    currently_parsing_noise_sample = False
    for i in range(len(rolling_std)):
        if rolling_std[i] < treshold:
            if not currently_parsing_noise_sample:
                currently_parsing_noise_sample = True
                noise_sample_start = i
        else:
            if currently_parsing_noise_sample:
                currently_parsing_noise_sample = False
                noise_sample_end = i
                # searching currently only for TARGET_SAMPLE_LENGTH long noise samples
                if (
                    noise_sample_end - noise_sample_start
                    >= float(TARGET_SAMPLE_LENGTH) * sampleRate
                ):
                    noise_samples.append(freq[noise_sample_start:noise_sample_end])



def main():

    if not os.path.exists('data/extracted_noises'):
        os.makedirs('data/extracted_noises')
    
    if not os.path.exists('data/noisy_audio'):
        print('Please provide the noisy audio in data/noisy_audio')
        return

    path_prefix = "data/noisy_audio/"
    soundfiles = [path_prefix + file for file in os.listdir("data/noisy_audio")]
    for file in tqdm(soundfiles):
        fr, sr = librosa.load(file)
        #This was left here, cause across all of the project I assumed that all of the audio I used has a sample rate of 22050. This was left here to see if I'm right.
        if sr != 22050:
            print(f'{sr}, {file}')
        extract_noise_samples(normalize_audio(fr), sr)
    print(len(noise_samples))
    with open("data/extracted_noises/noise_samples.pickle", "wb") as file:
        pickle.dump(noise_samples, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
