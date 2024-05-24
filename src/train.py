import torch
from GAN import GAN, DHMDataModule
import pytorch_lightning as pl




def main():
    
    directory = "data/training_data"
    dm = DHMDataModule(directory, batch_size=10)


    model = GAN(lr=1e-5)
    trainer = pl.Trainer(accelerator="auto",
        devices=1,
        max_epochs=50,
    )
    trainer.fit(model, dm)
    print('finished_training')
    path = './gan.pth'
    torch.save(model.state_dict(), path)
    
    

if __name__ == "__main__":
    main()