import numpy as np
import librosa
import pickle
import soundfile as sf


def apply_time_shift(noise_sample):
    shift_amount = np.random.randint(0, len(noise_sample))
    shifted_signal = np.roll(noise_sample, shift_amount)
    
    return shifted_signal

def add_perturbation(noise_sample):
    stft = librosa.stft(noise_sample)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Add Gaussian noise to the phase component
    phase_perturbation = np.random.normal(loc=0, scale=0.1, size=phase.shape)

    # Add perturbation to the phase
    phase += phase_perturbation

    # Reconstruct the complex STFT
    stft_with_perturbation = magnitude * np.exp(1j * phase)

    return librosa.istft(stft_with_perturbation)

def add_gain(noise, gain_dB):
    gain_linear = 10**(gain_dB / 20.0)
    
    noise_with_gain = [sample * gain_linear for sample in noise]
    
    return noise_with_gain

def noise_audio(noise_samples, clear_audio, sample_rate):
    noise = []
    # import pudb; pu.db
    while len(noise) < len(clear_audio):
        random_noise_sample = noise_samples[np.random.choice(len(noise_samples)-1)]
        modified_noise_sample = apply_time_shift(add_perturbation(random_noise_sample))
        # modified_noise_sample  = random_noise_sample
        overlap_length = int(0.2 * len(modified_noise_sample))
        if len(noise) == 0:
            noise.extend(modified_noise_sample)
        else:
            overlap_start_index = len(noise) - overlap_length
            if overlap_start_index < 0:
                overlap_start_index = 0
            overlap_end_index = overlap_start_index + overlap_length - 1
            for i, overlap_value in zip(range(overlap_start_index, overlap_end_index + 1), modified_noise_sample[:overlap_length]):
                noise[i] += overlap_value
            noise.extend(modified_noise_sample[overlap_length:])
    if len(noise) > len(clear_audio):
        noise = noise[:len(clear_audio)]

    noise_with_gain = add_gain(noise, 15)
    return [clear_audio[i] + noise_with_gain[i] for i in range(len(clear_audio))]

def main():
    with open('extracted_noises/noise_samples.pickle', 'rb') as file:
        noise_samples = pickle.load(file)
    fr, sr = librosa.load('clear_audio/classical.00000.wav')
    noise = noise_audio(noise_samples, fr, sr)
    sf.write('./noised_audio_made_by_me.flac', noise, sr, format='flac')


if __name__ == '__main__':
    main()