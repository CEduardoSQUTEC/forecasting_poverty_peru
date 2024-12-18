import lightning as L
from lightning.pytorch.tuner.tuning import Tuner

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import models
from torchvision.transforms import v2

transform = v2.Compose([
    v2.Resize(244),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

data = {
    split: datasets.CIFAR10(
        root='dataset/',
        download=True,
        transform=transform,
        train=split == 'train'
    ) for split in ['train', 'val']
}


class DenseNet121CIFAR10(L.LightningModule):
    def __init__(self, batch_size=1):
        super(DenseNet121CIFAR10, self).__init__()
        self.batch_size = batch_size
        densenet121 = models.densenet121()
        densenet121.classifier = nn.Linear(
            densenet121.classifier.in_features, 10
        )
        self.model = densenet121

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(dataset=data['train'], batch_size=self.batch_size, num_workers=4)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def val_dataloader(self):
        return DataLoader(dataset=data['val'], batch_size=self.batch_size, num_workers=4)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)


model = DenseNet121CIFAR10()

trainer = L.Trainer(
    strategy='auto',
    devices='auto',
    accelerator='auto',
    log_every_n_steps=8,
    max_epochs=4,
    min_epochs=1
)

if trainer.num_nodes == 1 and trainer.num_devices == 1:
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="power")

trainer.fit(model)
