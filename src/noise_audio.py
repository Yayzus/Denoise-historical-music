import numpy as np
import librosa
import pickle
import os
from tqdm import tqdm
import utils
from sklearn.model_selection import train_test_split
import soundfile as sf

save_idx = 0
snr = []

def noise_audio(noise_samples, clear_audio, sample_rate):
    global save_idx, snr
    noise = []
    #choosing a random noise sample to make noise from
    random_noise_segment = noise_samples[np.random.choice(len(noise_samples)-1)]

    while len(noise) < len(clear_audio):
        #applying a time shift with wraparound on the noise sample
        modified_noise_sample = utils.apply_time_shift(random_noise_segment)
        overlap_length = int(0.02 * len(modified_noise_sample))

        #for the first segment in the noise, we don't preform overlap and add
        if len(noise) == 0:
            noise.extend(modified_noise_sample)
        else:
            overlap_start_index = len(noise) - overlap_length
            overlap_end_index = len(noise) - 1
            for i, overlap_value in zip(range(overlap_start_index, overlap_end_index+1), modified_noise_sample[:overlap_length]):
                noise[i] = noise[i] + overlap_value
            noise.extend(modified_noise_sample[overlap_length:])
    if len(noise) > len(clear_audio):
        noise = noise[:len(clear_audio)]

    noise = utils.bandpass_filter(noise, 22050)
    
    noise_with_gain = utils.add_gain(noise)

    noised_audio = clear_audio + noise_with_gain

    snr.append(utils.snr(clear_audio, noised_audio))
    return utils.normalize_audio(noised_audio)


def main():
    # [[][]]
    training_data = []
    with open('data/extracted_noises/noise_samples.pickle', 'rb') as file:
        noise_samples = pickle.load(file)

    path_prefix = "data/clear_audio/"
    for file in tqdm(os.listdir('data/clear_audio')):
        fr, sr = librosa.load(path_prefix+file)
        #This was left here, cause across all of the project I assumed that all of the audio I used has a sample rate of 22050. This was left here to see if I'm right.
        if sr == 22050:
            fr = utils.normalize_audio(fr)
            start_idx = 0
            end_idx = 5*sr-1
            while end_idx <= len(fr):
                clear = fr[start_idx:end_idx]
                noisy = noise_audio(noise_samples, clear, sr)
                training_data.append([clear, noisy])
                start_idx += 5*sr
                end_idx += 5*sr     
    del(noise_samples)
    print(np.asarray(training_data).shape)
    train_data, test_data = train_test_split(training_data, test_size=0.2)

    print(f'train: {np.asarray(train_data).shape}')
    print(f'test: {np.asarray(test_data).shape}')
    print(f'snr: {np.sum(snr)/len(snr)}')

    sample_idx = 0
    for _ in range(20):
        random_sample = train_data[np.random.choice(len(train_data)-1)][1]
        sf.write(f'data/noisy_audio_samples/sample{sample_idx}.flac', random_sample, 22050, format='flac')
        sample_idx += 1

    #save training_data
    chunk_size = 10  # Adjust as needed
    with open("data/training_data.pickle", 'wb') as f:
        for i in range(0, len(train_data), chunk_size):
            pickle.dump(train_data[i:i+chunk_size], f)

    #save test data
    with open("data/test_data.pickle", 'wb') as f:
        for i in range(0, len(test_data), chunk_size):
            pickle.dump(test_data[i:i+chunk_size], f)

if __name__ == '__main__':
    main()