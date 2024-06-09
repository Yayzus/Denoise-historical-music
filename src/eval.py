import torch
import numpy as np
from GAN import GAN
import utils


# This script evaluates the trained model on the validation dataset

#Please change to the correct checkpoint file
model = GAN.load_from_checkpoint('lightning_logs/version_254/checkpoints/epoch=249-step=191500.ckpt')

#You can load the model from pth file, if you want
# model = GAN()
# model.load_state_dict(torch.load('gan.pth'))

model.eval()

snr_noisy = []
snr_denoised = []
snr = []

def inference(clear, noisy):
        #this function calculates the SNR of the noisy input, and the SNR of the denoised output of the model
        with torch.no_grad():
            prediction = model(torch.tensor(np.array([noisy]), device=model.device))
            snr_noisy.append(utils.snr(clear, noisy))
            snr_denoised.append(utils.snr(clear, prediction.cpu()))


validation_data = torch.load('validation_datamodule.pth')
for clear, noisy in validation_data:
        snr.append(utils.snr(clear, noisy))


print(f'val snr: {np.mean(snr)}')

#We divide the calidation set into 3 equal parts by the enties noise level
sorted_arguments = np.argsort(snr)
really_noisy_idxs = sorted_arguments[:int(len(sorted_arguments)/3)]
middle_indxs = sorted_arguments[int(len(sorted_arguments)/3):2*int(len(sorted_arguments)/3)]
not_that_noisy_idxs = sorted_arguments[2*int(len(sorted_arguments)/3):]

high_noise = []
middle_noise = []
low_noise = []

high_noise_snr = []
middle_noise_snr = []
low_noise_snr = []

#We create the 3 individual parts
for idx in really_noisy_idxs:
        high_noise.append(validation_data[idx])
        high_noise_snr.append(snr[idx])

print(f'high_noise snr: {np.mean(high_noise_snr)}')

for idx in middle_indxs:
        middle_noise.append(validation_data[idx])
        middle_noise_snr.append(snr[idx])

print(f'middle_noise snr {np.mean(middle_noise_snr)}')

for idx in not_that_noisy_idxs:
        low_noise.append(validation_data[idx])
        low_noise_snr.append(snr[idx])

print(f'low_noise snr {np.mean(low_noise_snr)}')



#we calculate the SNR difference of the noisy and denoised audios -> deltaSNR is the performance of our model
for clear, noisy in high_noise:
        inference(clear, noisy)

print(f'delta snr high_noise: {np.mean(snr_denoised) - np.mean(snr_noisy)}')  
snr_denoised = []
snr_noisy = []

for clear, noisy in middle_noise:
        inference(clear, noisy)

print(f'delta snr middle_noise: {np.mean(snr_denoised) - np.mean(snr_noisy)}')  
snr_denoised = []
snr_noisy = []

for clear, noisy in low_noise:
        inference(clear, noisy)

print(f'delta snr low_noise: {np.mean(snr_denoised) - np.mean(snr_noisy)}')  
snr_denoised = []
snr_noisy = []


