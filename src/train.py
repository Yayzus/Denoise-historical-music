import torch
from GAN import GAN, DHMDataModule
import pytorch_lightning as pl




def main():
    
    directory = "data/training_data"
    dm = DHMDataModule(directory, batch_size=8)


    model = GAN(lr_g=5e-5, lr_d=5e-5,b1=0.9, b2=0.999)
    trainer = pl.Trainer(accelerator="auto",
        devices=1,
        max_epochs=20,
    )
    trainer.fit(model, dm)
    print('finished_training')
    path = './gan.pth'
    torch.save(model.state_dict(), path)
    
    

if __name__ == "__main__":
    main()