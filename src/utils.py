import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import librosa
import torch

def normalize_audio(audio, target_peak=0.99):
    #Normalizing an audio to have a max amplitude at target_peak
    peak = np.max(np.abs(audio))
    normalization_factor = target_peak / peak
    normalized_audio = audio * normalization_factor
    return normalized_audio

def bandpass_filter(data, sample_rate):
    nyquist = 0.5 * sample_rate
    lowcut = np.random.uniform(50, 150)
    highcut = np.random.uniform(5000, 10000)
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, data)

def apply_time_shift(noise_sample):
    #Applying time shift with wraparound
    shift_amount = np.random.randint(0, len(noise_sample))
    shifted_signal = np.roll(noise_sample, shift_amount)
    
    return shifted_signal

def add_gain(noise):
    #Adding a random gain between [15, 30]dB to a noise segment 
    gain_dB = np.random.uniform(15, 30)
    gain_linear = int(10**(gain_dB / 20.0))
    
    noise_with_gain = noise*gain_linear
    
    return noise_with_gain

def plot_sample(noise, ax = None):
    #plotting an audio sample.
    times = np.linspace(0, len(noise)/22050, num=len(noise))
    if ax == None:
        plt.plot(times, noise)
        plt.show()
    else:
        ax.plot(times, noise)

def convert_audio_to_image(x):
    #converting waveform audio to 2 channel image (waveform -> model input)
    stft = librosa.stft(np.asarray(x), n_fft=1024, hop_length=512)
    real = np.real(stft)
    imaginary = np.imag(stft)

    img =  np.stack((real, imaginary), axis=0)

    return img

def convert_image_to_audio(image):
    #converting a 2 channel image to waveform audio (model output -> waveform)
    real_part = image[0]
    imaginary_part = image[1]

    complex_matrix = real_part + 1j * imaginary_part
    #If our image is a tensor, and it's on the gpu, we have to move it to the cpu.
    if isinstance(image, torch.Tensor):
        return librosa.istft(np.asarray(complex_matrix.cpu()), n_fft=1024, hop_length=512)
    else:
        return librosa.istft(np.asarray(complex_matrix), n_fft=1024, hop_length=512)

def snr(clear, noisy, spectogram=False):
    #Calculating Signal to Noise ratio
    #It works both for waveform audio and 2 channel image 
    if spectogram:
        clear_audio = np.array(convert_image_to_audio(clear.cpu()))
        noisy_audio = np.array(convert_image_to_audio(noisy.cpu()))
    else:
        clear_audio = np.array(clear)
        noisy_audio = np.array(noisy)

    signal_power = np.mean(clear_audio ** 2)
    
    # Calculate the noise power
    noise = noisy_audio - clear_audio
    noise_power = np.mean(noise ** 2)
    
    # Calculate the SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

