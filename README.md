This is the project for  my dissertation thesis. It is based on https://arxiv.org/pdf/2008.02027

how the project works:
1. extract_noise.py: Extracts noise samples from the audio files in data/noisy/audio and pickle dumps it into data/extracted_noise/noise_samples.pickle
2. noise_audio.py: creates <clear, noisy> pairs from the files in data/clear_audio/ using the noise samples mentioned above (waveform audio). Saves it to data/training_data.pickle
3. preprocess.py: Transformes the above mentioned training data into the time and frequency domain (STFT: waveform -> 2 channel image). Saves every <clear, noisy> pair into a seperate file (Made a custom Dataset from wich the DataLoader loads the data, for memory saving purposes, only the pairs that are in use are loaded in from the hard disk)
4. GAN.py: This file contains everything needed for training the GAN: The GAN architecture, DataModule, the above mentioned custom DataSet.
5. train.py: a short script that trains a GAN and saves it to gan.pth
6. plot.py: use this after training to see how the metrics change over the epochs.
7. evaluation.py: script to evaluate the model on the validation set. It can load the model from gan.pth or from lightning checkpoint 
