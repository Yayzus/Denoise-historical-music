import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import pandas as pd
import pickle

Q = 0.005 #45%
T = 0.1 # window size in s
TARGET_SAMPLE_LENGTH = 0.1 #minimum length of desired noise samples is s


def extract_noise_samples(freq, sampleRate):
    #window size is 100 ms, which means we get sampleRate/10 samples in our window
    window = int(sampleRate*T)
    data = pd.Series(freq)
    rolling_std = data.rolling(window).std()
    sorted_std_indexes = rolling_std.argsort()
    print(f'{rolling_std[sorted_std_indexes[10000]]}, {rolling_std[sorted_std_indexes[20000]]}')
    #we estimate an adaptive threshold τ based on the q-th quantile (percentile) of the standard deviations
    treshold = rolling_std[sorted_std_indexes[int(len(rolling_std)*float(Q))]]
    print(f'treshold: {treshold}')


    noise_samples = []
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
                #searching currently only for noise samples
                if(noise_sample_end - noise_sample_start >= float(TARGET_SAMPLE_LENGTH)*sampleRate):
                    noise_samples.append((noise_sample_start, noise_sample_end))
    return [freq[noise_sample_start:noise_sample_end] for noise_sample_start, noise_sample_end in noise_samples]

def main():

    fr, sr = librosa.load('beka-3615-31034.flac')
    print(f'sample rate: {sr}')
    noise_samples = extract_noise_samples(fr, sr)
    with open('extracted_noises/noise_samples.pickle', 'wb') as file:
        pickle.dump(noise_samples, file, protocol=pickle.HIGHEST_PROTOCOL)
    # for i in range(len(noise_samples)):
    #     print(noise_samples[i])
    #     noise_sample_start, noise_sample_end = nois
    # e_samples[i]
    #     path = f'./extracted_noises/noise_{i}.flac'
    #     sf.write(path, fr[noise_sample_start:noise_sample_end], sr, format='flac')

if __name__ == '__main__':
    main()