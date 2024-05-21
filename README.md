This is part of my dissertation thesis. It is based on https://arxiv.org/pdf/2008.02027

how the project works:
1. download_audio.py: Downloads the noisy audio files from the Public Domain Project into data/noisy_audio/. It has two parts, first we scrape the links (rn it's commented) and then downloading the audio files .
2. extract_noise.py: Extracts noise samples from the audio files in data/noisy/audio and pickle dumps it into data/extracted_noise/noise_samples.pickle
3. noise_audio.py: creates <clear, noisy> pairs from the files in data/clear_audio/ using the noise samples mentioned above (waveform audio). Saves it to data/training_data.pickle
4. preprocess.py: Transformes the above mentioned training data into the time and frequency domain (STFT: waveform -> 2 channel image). Saves every <clear, noisy> pair into a seperate file (Made a custom Dataset from wich the DataLoader loads the data, for memory saving purposes, only the pairs that are in use are loaded in from the hard disk)
5. GAN.py: This file contains everything needed for training the GAN: The GAN architecture, DataModule, the above mentioned custom DataSet.
6. train.py: a short script that trains a GAN and saves it to gan.pth
