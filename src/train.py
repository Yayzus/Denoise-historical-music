from src.GAN import Generator, Discriminator
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch
from src.GAN import GAN, DHMDataModule
import pytorch_lightning as pl





def main():
    
    directory = "data/training_data"
    dm = DHMDataModule(directory, batch_size=3)


    model = GAN(lr=0.02)
    trainer = pl.Trainer(accelerator="auto",
        devices=1,
        max_epochs=150,
    )
    trainer.fit(model, dm)
    print('finished_training')
    path = './gan.pth'
    torch.save(model.state_dict(), path)
    
    

if __name__ == "__main__":
    main()