from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
from torchvision.transforms import v2
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from PIL import Image

from forecasting_poverty_peru.config import MODELS_PATH, PROCESSED_PATH
from forecasting_poverty_peru.modeling.satmae_pp.fmow_rgb.models_vit import vit_large_patch16

SIENAHO_PATH = PROCESSED_PATH / 'sienaho_rgb'

CHECKPOINT_FILEPATH = (
    MODELS_PATH /
    'satmae_pp' /
    'fmow_rgb' /
    'checkpoints' /
    'ViT-L_finetune.pth'
)


class LitFinetuneViTL(L.LightningModule):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.vit(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 7e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


model = vit_large_patch16(
    num_classes=62,
    img_size=(224, 224),
    patch_size=(16, 16),
    in_chans=3
)

checkpoint = torch.load(CHECKPOINT_FILEPATH)
checkpoint['model'] = {
    k.replace('fc_norm', 'norm'): v for k, v in checkpoint['model'].items()
}

model.load_state_dict(checkpoint['model'])

model.head = nn.Linear(
    in_features=1024,
    out_features=2
)

transform = v2.Compose([
    v2.RandomCrop((224, 224)),
    v2.PILToTensor()
])

dataset = DatasetFolder(
    root=SIENAHO_PATH,
    loader=Image.open,
    extensions=['.tif'],
    transform=transform,
    target_transform=lambda x: x == 'adequate'
)

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

seed = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=seed
)

lit_model = LitFinetuneViTL(model)

logger = TensorBoardLogger("logs", name="poverty_forecasting")

trainer = L.Trainer(
    max_epochs=5,
    logger=logger,
)

trainer.fit(lit_model, DataLoader(train_dataset), DataLoader(val_dataset))

trainer.test(lit_model, DataLoader(test_dataset))
