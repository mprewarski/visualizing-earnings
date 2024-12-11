
import os
from typing import Any
from torch import optim, nn, utils, Tensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
import torchvision
import torchvision.transforms as tt
import timm
import torch.nn.functional as F
from pathlib import Path
from torchmetrics import Accuracy, ConfusionMatrix
import torch

data_dir = Path("/Users/marcusprewarski/projects/stock-models/earn_train_data")
batch_size = 64
n_loaders = 4


train_tt= tt.Compose([
    tt.Resize((48,48)),
    tt.ToTensor(),
    tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],inplace=True)
])
val_tt= tt.Compose([
    tt.Resize((48,48)),
    tt.ToTensor(),
    tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],inplace=True)
])

class MyClassifier(L.LightningModule):
    def __init__(self, arch:str, num_classes:int, lr=1e-3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.arch = arch
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.model = timm.create_model(arch, pretrained=True, num_classes = num_classes)

    def forward(self, x:Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.model(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), self.lr)
        return opt

    def training_step(self, batch, batch_idx):
        x,y = batch
        z = self.model(x)
        loss = self.criterion(z,y)
        batch_acc = self.train_acc(z, y)
        acc = self.train_acc.compute()
        self.log_dict({"train acc": acc, "loss": loss}, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y=batch
        z= self.model(x)
        loss= self.criterion(z,y)
        batch_acc = self.val_acc(z, y)
        acc = self.val_acc.compute()
        self.log_dict({"val_acc": acc, "loss": loss}, prog_bar=True, on_step=False, on_epoch=True )
        return loss
    
    def on_validation_epoch_start(self):
        self.val_acc.reset()
        self.train_acc.reset()


# saves top-K checkpoints based on "val_acc" metric
# the monitor metric is the one that is used to determine the best checkpoint, 
# must be a metric that is logged in the validation_step
checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    monitor="val_acc",
    mode="max",
    filename="chk--{epoch:02d}-{val_acc:.2f}"
)

if __name__ == "__main__":
    # create datasets and dataloaders
    print("Creating datasets and dataloaders")
    train_dataset = ImageFolder(data_dir/'train', transform=train_tt)
    val_dataset = ImageFolder(data_dir/'val', transform=val_tt)
    train_loader = utils.data.DataLoader( train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders, persistent_workers=True)
    val_loader = utils.data.DataLoader( val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders, persistent_workers=True)

    # create model
    print("Creating model")
    num_classes = len(train_dataset.classes)
    model = MyClassifier("resnet18", num_classes, lr=0.00001)

    # create trainer and fit
    print("Creating trainer")
    trainer = L.Trainer(max_epochs=20, callbacks=[checkpoint_callback])
    run_lr_finder = False

    if run_lr_finder:
        print("Running LR finder")
        tuner = Tuner(trainer=trainer)
        lr_finder = tuner.lr_find(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, min_lr=1e-7, max_lr=0.1 )
        lr_finder.plot(suggest=True)
    else:
        print("Fitting model")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
