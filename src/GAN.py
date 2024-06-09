import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset
import os
from utils import snr
from torch.nn.utils.parametrizations import weight_norm
# from torchmetrics.classification import BinaryHingeLoss


class E_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, weight_normalization = False, activation = 'ELU'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        if stride == (1, 2):
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 4), stride=stride, padding=1
            )
        else:
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=(4, 4), stride=stride, padding = 1
            )
        if(weight_normalization):
            weight_norm(self.conv1)
            weight_norm(self.conv2)
        if activation == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU()


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
                in_channels, out_channels, kernel_size=(3, 4), stride=stride, padding=1
            )
        else:
            self.transconv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=stride, padding = 1
            )
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1)),
                nn.Upsample(scale_factor=stride, mode="nearest")
            )
        else:
            self.shortcut = nn.Upsample(scale_factor=stride, mode="nearest")

        weight_norm(self.transconv)
        weight_norm(self.conv)
        self.activation = nn.ELU()

    def forward(self, x):
        shortcut_state = x

        x = self.transconv(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.activation(x)
        shortcut_state = self.shortcut(shortcut_state)

        return x + shortcut_state

class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(2, 32, kernel_size=7, padding=3))
        self.conv2 = weight_norm(nn.Conv2d(512, 128, kernel_size=3, padding=1))
        self.conv3 = weight_norm(nn.Conv2d(128, 512, kernel_size=3, padding=1))
        self.conv4 = weight_norm(nn.Conv2d(32, 2, kernel_size=7, padding=3))

        self.eblock1 = E_block(32, 64, stride=(1, 2), weight_normalization=True)
        self.eblock2 = E_block(64, 64, stride=(1, 2), weight_normalization=True)
        self.eblock3 = E_block(64, 128, stride=(2, 2), weight_normalization=True)
        self.eblock4 = E_block(128, 128, stride=(1, 2), weight_normalization=True)
        self.eblock5 = E_block(128, 256, stride=(2, 2), weight_normalization=True)
        self.eblock6 = E_block(256, 512, stride=(2, 2), weight_normalization=True)

        self.dblock1 = D_block(512, 256, stride=(2, 2))
        self.dblock2 = D_block(256, 128, stride=(2, 2))
        self.dblock3 = D_block(128, 128, stride=(1, 2))
        self.dblock4 = D_block(128, 64, stride=(2, 2))
        self.dblock5 = D_block(64, 64, stride=(1, 2))
        self.dblock6 = D_block(64, 32, stride=(1, 2))

        self.activation = nn.ELU()
        

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
        if skip7.shape != x.shape:
            x = self.correct_dimmensions(skip7, x)
        x = self.activation(x + skip7)

        x = self.dblock1(x)
        if skip6.shape != x.shape:
            x = self.correct_dimmensions(skip6, x)
        x = self.activation(x + skip6)

        x = self.dblock2(x)
        if skip5.shape != x.shape:
            x = self.correct_dimmensions(skip5, x)
        x = self.activation(x + skip5)


        x = self.dblock3(x) 
        if skip4.shape != x.shape:
            x = self.correct_dimmensions(skip4, x)
        x = self.activation(x + skip4)

        x = self.dblock4(x) 
        if skip3.shape != x.shape:
            x = self.correct_dimmensions(skip3, x)
        x = self.activation(x + skip3)

        x = self.dblock5(x)
        if skip2.shape != x.shape:
            x = self.correct_dimmensions(skip2, x)
        x = self.activation(x + skip2)

        x = self.dblock6(x)
        if skip1.shape != x.shape:
            x = self.correct_dimmensions(skip1, x)
        x = self.activation(x + skip1)
        x = self.conv4(x)
        
        return x


class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)

        self.eblock1 = E_block(32, 64, stride=(1, 2), activation='LeakyReLU')
        self.eblock2 = E_block(64, 64, stride=(1, 2), activation='LeakyReLU')
        self.eblock3 = E_block(64, 128, stride=(2, 2), activation='LeakyReLU')
        self.eblock4 = E_block(128, 128, stride=(1, 2), activation='LeakyReLU')
        self.eblock5 = E_block(128, 256, stride=(2, 2), activation='LeakyReLU')
        self.eblock6 = E_block(256, 512, stride=(2, 2), activation='LeakyReLU')

        self.activation = nn.LeakyReLU(negative_slope=0.3)
    
    def forward(self, x):
        x = self.activation(F.layer_norm(self.conv1(x), [32,513,216]))
        x = self.activation(F.layer_norm(self.eblock1(x),[64, 513, 108]))
    
        x = self.activation(F.layer_norm(self.eblock2(x), [64, 513, 54]))
        x = self.activation(F.layer_norm(self.eblock3(x), [128, 256, 27]))
        x = self.activation(F.layer_norm(self.eblock4(x), [128, 256, 13]))

        x = self.activation(F.layer_norm(self.eblock5(x), [256, 128, 6]))

        x = self.activation(F.layer_norm(self.eblock6(x), [512, 64, 3])
)
        x = self.conv2(x)

        return x

class DHMDataSet(Dataset):
    def __init__(self, directory):
        super().__init__()
        self.files = os.listdir(directory)
        self.prefix = directory

    def __getitem__(self, index):
        clear_img, noisy_img = np.float32(np.load(f'{self.prefix}/{self.files[index]}'))
        return (torch.tensor(clear_img), torch.tensor(noisy_img))
    
    def __len__(self):
        return len(self.files)

class DHMDataModule(pl.LightningDataModule):
    def __init__(self, directory, batch_size=7, num_workers=16) -> None:
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loaded_from_file = False

    def prepare_data(self):
        if os.path.exists('training_datamodule.pth'):
            self.train_data = torch.load('training_datamodule.pth')
            self.loaded_from_file = True
        if os.path.exists('validation_datamodule.pth'):
            self.val_data = torch.load('validation_datamodule.pth')
            self.loaded_from_file = True

    def setup(self, stage = None):
        if not self.loaded_from_file:

            images = DHMDataSet(self.directory)
            self.train_data, self.val_data = random_split(images, [0.8, 0.2])

            torch.save(self.train_data, 'training_datamodule.pth')
            torch.save(self.val_data, 'validation_datamodule.pth')


    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        pass

class GAN(pl.LightningModule):
    def __init__(self, lr_g=0.0001, lr_d = 0.0001, b1 = 0.5, b2 = 0.9) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.g_val_loss = []
        self.d_val_clear_loss = []
        self.d_val_noisy_loss = []
        self.d_val_loss = []
        self.snr = []
        self.g_rec_loss = []

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, input, label):
        zeros = torch.zeros_like(input)
        return torch.mean(torch.mean(torch.max(zeros, 1-label*input), dim=(1,2,3)))
    
    def signal_to_noise(self, clean_batch, denoised_batch):
        snr_list = []
        for clean, noisy in zip(clean_batch, denoised_batch):
            snr_list.append(snr(clean, noisy, spectogram = True))
        return snr_list
    
    def reconstruction_loss(self, denoised, clear_data):
        #reconstruction loss is L1 loss
        complex_denoised = denoised[:,0,:,:] + 1j*denoised[:,1,:,:]
        complex_clear = clear_data[:,0,:,:] + 1j*clear_data[:,1,:,:]
        
        return nn.L1Loss()(complex_denoised, complex_clear)

        # abs_sum = torch.sum(torch.abs(clear_data-denoised))
        # return abs_sum/denoised.size(3)

    def training_step(self, batch):
        clear_imgs, noisy_imgs = batch

        denoised_imgs = self(noisy_imgs)

        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()

        #train the generator

        g_loss_rec = self.reconstruction_loss(denoised_imgs, clear_imgs)
        
        #we test if the generator can fool the discriminator
        y_hat = self.discriminator(denoised_imgs)


        g_loss_adv = self.adversarial_loss(y_hat, 1)

        g_loss = g_loss_rec + 0.01*g_loss_adv

        self.manual_backward(g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        #train the discriminator
        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()

        #how well can the discriminator detect clear audio

        y_hat = self.discriminator(clear_imgs)

        clear_loss = self.adversarial_loss(y_hat, 1)

        #how well can the discriminator detect clear audio made by the generator

        y_hat = self.discriminator(denoised_imgs.detach())

        noisy_loss = self.adversarial_loss(y_hat, -1)


        #the discriminator loss is the sum of these
        d_loss = clear_loss + noisy_loss
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)




    def configure_optimizers(self):
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        opt_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1,b2))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(b1,b2))

        return [opt_g, opt_d], []
    
    def validation_step(self, batch):

        clear_imgs, noisy_imgs = batch
        denoised = self(noisy_imgs).detach()
        predictions_for_denoised = self.discriminator(denoised).detach()

        # device = 'cuda' if noisy_imgs.is_cuda else 'cpu'


        self.g_val_loss.append(self.adversarial_loss(predictions_for_denoised, 1))
        clear_loss = self.adversarial_loss(self.discriminator(clear_imgs.detach()).detach(), 1)
        noisy_loss = self.adversarial_loss(predictions_for_denoised, -1)
        d_loss = clear_loss+noisy_loss
        g_rec_loss = self.reconstruction_loss(denoised, clear_imgs)

        self.d_val_loss.append(d_loss)
        self.d_val_clear_loss.append(clear_loss)
        self.d_val_noisy_loss.append(noisy_loss)
        self.snr.extend(self.signal_to_noise(clear_imgs, denoised))
        self.g_rec_loss.append(g_rec_loss)

    def on_validation_epoch_end(self):
        g_loss_val = sum(self.g_val_loss)/len(self.g_val_loss)
        self.log("g_adv_loss", g_loss_val, prog_bar=True)

        d_loss_val = sum(self.d_val_loss)/len(self.d_val_loss)
        self.log("d_adv_loss", d_loss_val, prog_bar=True)

        d_clear_loss_val = sum(self.d_val_clear_loss)/len(self.d_val_clear_loss)
        self.log("d_clear_loss_val", d_clear_loss_val, prog_bar=False)

        d_noisy_loss_val = sum(self.d_val_noisy_loss)/len(self.d_val_noisy_loss)
        self.log("d_noisy_loss_val", d_noisy_loss_val, prog_bar=False)

        g_rec_loss = sum(self.g_rec_loss)/len(self.g_rec_loss)
        self.log("g_rec_loss", g_rec_loss, prog_bar=True)

        self.log("snr", sum(self.snr)/len(self.snr), prog_bar=True)
        self.g_val_loss = []
        self.d_val_clear_loss = []
        self.d_val_noisy_loss = []
        self.d_val_loss = []
        self.snr = []
        self.g_rec_loss = []