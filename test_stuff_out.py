import librosa
import torch.nn as nn
import torch
import numpy as np
from GAN import GAN, DHMDataModule, Generator
import pickle
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl

# fr, sr = librosa.load('data/clear_audio/classical.00000.wav')

# print(f'Frequeancies shape: {fr.shape}')
# print(f'Sample rate: {sr}')

# D = librosa.stft(fr)
# print(D.shape)
# # print(librosa.amplitude_to_db(D, ref=np.max))
# X = librosa.amplitude_to_db(D, ref=np.max)

# downsample_frequency_conv = nn.Conv2d(in_channels=1, 
#                                       out_channels=1, 
#                                       kernel_size=(1, 3),  # Kernel size along time (1) and frequency (3)
#                                       stride=(1, 2),       # Stride along time (1) and frequency (2)
#                                       padding=(0, 1))      # Padding to maintain output size

# block = E_block(1,1,(2,2))
# # Apply the convolutional layer to the input spectrogram
# output_spectrogram = block(torch.tensor([X]))
# print(output_spectrogram.shape)
# print(X.shape)

# import matplotlib.pyplot as plt
# fig, (subpltL, subpltR) = plt.subplots(1,2, figsize=(50, 75))
# img = librosa.display.specshow(X,
#                                y_axis='log', x_axis='time', ax=subpltL)
# subpltL.set_title('Power spectrogram')
# fig.colorbar(img, ax=subpltL, format="%+2.0f dB")

# img2 = librosa.display.specshow(output_spectrogram[0].detach().numpy(),
#                                y_axis='log', x_axis='time', ax=subpltR)
# subpltL.set_title('Power spectrogram')
# fig.colorbar(img2, ax=subpltR, format="%+2.0f dB")

# plt.show()



# print(X.shape)
# upsample = nn.Upsample(scale_factor=(1,2), mode='nearest')
# output_spectrogram2 = upsample(torch.tensor(np.array([[X]])))
# print(output_spectrogram2.shape)

# ize = torch.tensor([X])
# print(ize.shape)
# ize2 = ize.view(1, *ize.shape)
# print(ize2.shape)

# real_part = np.real(D)
# imaginary_part = np.imag(D)

# image_2_channel = np.stack((real_part, imaginary_part), axis=0)

# print(image_2_channel.shape)

# block2 = Generator()
# output_spectrogram2 = block2(torch.tensor(image_2_channel))
# print(output_spectrogram2.shape)

# print(np.empty(0))
# with open('data/extracted_noises/noise_samples.pickle', 'rb') as file:
#         noise_samples = pickle.load(file)

file = "data/training_data.pickle"

dm = DHMDataModule(file)
model = GAN()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, dm)
print('finished_training')
path = './gan.pth'
torch.save(model.state_dict(), path)
# dm.setup()
# model = Generator()
# for i, (batch_x, batch_y) in enumerate(dm.train_dataloader()):
#     print(f"Batch {i}: input shape {batch_x.shape}, label shape {batch_y.shape}")
#     model(batch_y)



