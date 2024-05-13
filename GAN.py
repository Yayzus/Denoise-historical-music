from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import librosa
from torch.utils.data import DataLoader, random_split, TensorDataset


class E_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        if stride == (1, 2):
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 4), stride=stride, padding=1
            )
        else:
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=stride, padding=1
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        return x


class D_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__()
        if stride == (1, 2):
            self.transconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=(3, 4), stride=stride
            )
        else:
            self.transconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=stride
            )
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = nn.Upsample(scale_factor=stride, mode="nearest")
        self.activation = nn.ReLU()

    def forward(self, x):
        shortcut_state = x

        x = self.transconv(x)
        x = self.activation(x)
        x = self.conv(x)

        shortcut_state = self.shortcut(shortcut_state)

        return x + shortcut_state


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 2, kernel_size=7, padding=3)

        self.eblock1 = E_block(32, 64, stride=(1, 2))
        self.eblock2 = E_block(64, 64, stride=(1, 2))
        self.eblock3 = E_block(64, 128, stride=(2, 2))
        self.eblock4 = E_block(128, 128, stride=(1, 2))
        self.eblock5 = E_block(128, 256, stride=(2, 2))
        self.eblock6 = E_block(256, 512, stride=(2, 2))

        self.dblock1 = D_block(512, 256, stride=(2, 2))
        self.dblock2 = D_block(256, 128, stride=(2, 2))
        self.dblock3 = D_block(128, 128, stride=(1, 2))
        self.dblock4 = D_block(128, 64, stride=(2, 2))
        self.dblock5 = D_block(64, 64, stride=(1, 2))
        self.dblock6 = D_block(64, 32, stride=(1, 2))

        self.activation = nn.ReLU()

    def correct_dimmensions(self, skip, x):

        frequency_domain_diff = np.abs(skip.shape[2] - x.shape[2])
        time_domain_diff = np.abs(skip.shape[3] - x.shape[3])
        cor_dim = F.pad(
            x, (0, time_domain_diff, 0, frequency_domain_diff), mode="constant", value=0
        )

        return cor_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        skip1 = x

        x = self.eblock1(x)
        x = self.activation(x)
        skip2 = x

        x = self.eblock2(x)
        x = self.activation(x)
        skip3 = x

        x = self.eblock3(x)
        x = self.activation(x)
        skip4 = x

        x = self.eblock4(x)
        x = self.activation(x)
        skip5 = x

        x = self.eblock5(x)
        x = self.activation(x)
        skip6 = x

        x = self.eblock6(x)
        x = self.activation(x)
        skip7 = x

        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)

        if skip7.shape != x.shape:
            x = self.correct_dimmensions(skip7, x)
        x = self.dblock1(x + skip7)
        x = self.activation(x)

        if skip6.shape != x.shape:
            x = self.correct_dimmensions(skip6, x)
        x = self.dblock2(x + skip6)
        x = self.activation(x)

        if skip5.shape != x.shape:
            x = self.correct_dimmensions(skip5, x)
        x = self.dblock3(x + skip5)
        x = self.activation(x)

        if skip4.shape != x.shape:
            x = self.correct_dimmensions(skip4, x)
        x = self.dblock4(x + skip4)
        x = self.activation(x)

        if skip3.shape != x.shape:
            x = self.correct_dimmensions(skip3, x)
        x = self.dblock5(x + skip3)
        x = self.activation(x)

        if skip2.shape != x.shape:
            x = self.correct_dimmensions(skip2, x)
        x = self.dblock6(x + skip2)
        x = self.activation(x)

        if skip1.shape != x.shape:
            x = self.correct_dimmensions(skip1, x)
        x = self.conv4(x + skip1)

        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

        self.eblock1 = E_block(32, 64, stride=(1, 2))
        self.eblock2 = E_block(64, 64, stride=(1, 2))
        self.eblock3 = E_block(64, 128, stride=(2, 2))
        self.eblock4 = E_block(128, 128, stride=(1, 2))
        self.eblock5 = E_block(128, 256, stride=(2, 2))
        self.eblock6 = E_block(256, 512, stride=(2, 2))

        self.fc1 = nn.Linear(128 * 128 * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        x = self.eblock1(x)
        x = self.activation(x)

        x = self.eblock2(x)
        x = self.activation(x)

        x = self.eblock3(x)
        x = self.activation(x)

        x = self.eblock4(x)
        x = self.activation(x)

        x = self.eblock5(x)
        x = self.activation(x)

        x = self.eblock6(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)

        x = x.view(-1, 128 * 128 * 6)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)

        return torch.sigmoid(x)


class DHMDataModule(pl.LightningDataModule):
    def __init__(self, input_file, batch_size=4, num_workers=4) -> None:
        super().__init__()
        self.data_file = input_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def read_list_from_file(self, filename):
        result = []
        with open(filename, "rb") as f:
            while True:
                try:
                    chunk = pickle.load(f)
                    result.extend(chunk)
                except EOFError:
                    break
        return result

    def convert_matrix_to_2_channel_image(self, x):
        real_part = np.real(x)
        imaginary_part = np.imag(x)

        image_2_channel = np.stack((real_part, imaginary_part), axis=0)

        return image_2_channel

    def setup(self, stage = None):
        full_data = self.read_list_from_file(self.data_file)
        clear_audio = [audio[0] for audio in full_data]
        noisy_audio = [audio[1] for audio in full_data]
        del full_data

        for i in range(0, len(clear_audio)):
            clear_audio[i] = self.convert_matrix_to_2_channel_image(
                librosa.stft(np.asarray(clear_audio[i]))
            )
            noisy_audio[i] = self.convert_matrix_to_2_channel_image(
                librosa.stft(np.asarray(noisy_audio[i]))
            )

        clear_audio = np.array(clear_audio, dtype=np.float32)
        noisy_audio = np.array(noisy_audio, dtype=np.float32)

        imgs = TensorDataset(torch.tensor(noisy_audio), torch.tensor(clear_audio))
        first_split, self.test_data = random_split(imgs, [0.8, 0.2])
        self.train_data, self.val_data = random_split(first_split, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )


class GAN(pl.LightningModule):
    def __init__(self, lr=0.0002) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch):
        noisy_imgs, clear_imgs = batch

        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_g)
        generated_imgs = self(noisy_imgs)

        #train the generator

        #ground thruth
        y = torch.ones(noisy_imgs.size(0), 1)
        y.type_as(noisy_imgs)

        #we test if the generator can fool the discriminator
        y_hat = self.discriminator(generated_imgs)
        g_loss = self.adversarial_loss(y_hat, y)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        #train the discriminator
        self.toggle_optimizer(optimizer_d)

        #how well can the discriminator detect clear audio
        y = torch.ones(clear_imgs.size(0), 1)
        y.type_as(clear_imgs)

        y_hat = self.discriminator(clear_imgs)

        clear_loss = self.adversarial_loss(y_hat, y)

        #how well can the discriminator detect noisy audio
        y = torch.zeros(noisy_imgs.size(0), 1)
        y.type_as(noisy_imgs)

        y_hat = self.discriminator(noisy_imgs)

        noisy_loss = self.adversarial_loss(y_hat, y)


        #the discriminator loss is the average of these
        d_loss = (clear_loss + noisy_loss)/2
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)




    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr)

        return [opt_g, opt_d], []
