import torch
from GAN import GAN, DHMDataModule
import pytorch_lightning as pl




def main():
    
    directory = "data/training_data"

    #The datamodule loads in the training and validation data from data/training_datamodule.pth and data/validation_datamodule.pth
    #delete those files if you want new training and validation datasets
    dm = DHMDataModule(directory, batch_size=6, num_workers=12)


    model = GAN(lr_g=4e-6, lr_d=2e-6,b1=0.9, b2=0.999)
    trainer = pl.Trainer(accelerator="auto",
        devices=1,
        max_epochs=250,
    )
    trainer.fit(model, dm)
    print('finished_training')
    path = './gan.pth'
    torch.save(model.state_dict(), path)
    
    

if __name__ == "__main__":
    main()