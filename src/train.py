from torchvision import models, transforms
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from data_loader import CatDogDataLoader
from torchmetrics import Accuracy
from torch.nn import functional as F
from torch import nn
import torch
import wrapper
from pathlib import Path

current_dir = Path(__file__).absolute().parent.parent
data_dir = current_dir.joinpath('data/cat-dogs')
src_dir = current_dir.joinpath('src')
model_dir = current_dir.joinpath('models/')
criterion = nn.CrossEntropyLoss()
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'


class CatDogTrainer(pl.LightningModule):
    def __init__(self, input_shape: int, crit: str, batch_size: int = 1, learning_rate: float = 0.4):
        super().__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.criterion = crit
        self.num_classes = 2
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="binary", num_classes=2)
        use_pretrain = True
        self.model = models.vgg16(pretrained=use_pretrain)
        self.model.classifier[6] = nn.Linear(in_features=self.model.classifier[6].in_features,
                                             out_features=self.num_classes)
        params_to_update = []
        update_params = ['classifier.6.weight', 'classifier.6.bias']
        for name, params in self.model.named_parameters():
            if name in update_params:
                params.requires_grad = True
                params_to_update.append(params)
            else:
                params.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch: int, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_f1', acc, on_epoch=True)
        return loss

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    # test loop
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        return loss

    # optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    dataset = CatDogDataLoader(data_dir)
    print(len(dataset))
    size_test = int(len(dataset) * 0.20)
    size_train = len(dataset) - size_test
    train, val = random_split(dataset, [size_train, size_test])
    train_dataloader = DataLoader(train, batch_size=1, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val, batch_size=1, shuffle=True, num_workers=1)
    model = CatDogTrainer((3, 224, 224), batch_size=1, crit=criterion)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = "cat-dog-classifier"
    mlflow.set_experiment(experiment_name)
    ml_logger = MLFlowLogger(experiment_name=experiment_name, artifact_location=str(model_dir),
                             tracking_uri=mlflow.get_tracking_uri(),
                             )
    trainer = Trainer(accelerator='cpu', max_epochs=1, logger=ml_logger)
    trainer.fit(model, train_dataloader)
    trainer.validate(model=model, dataloaders=val_dataloader)
    train_model_pth = model_dir.joinpath('model.pt')
    torch.save(model, train_model_pth)
    artifacts = {
        "models": str(train_model_pth)
    }
    mlflow.end_run()
    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "hswat")
        mlflow.set_tag("algo", "vgg16")
        model = mlflow.pyfunc.log_model(artifact_path="models", artifacts=artifacts, python_model=wrapper.TestWrapper(),
                                        code_path=[
                                            'C:\\Users\\hswat\\PycharmProjects\\mlflow-prefect-pytorch\\src\\wrapper.py'])
        print(model.model_uri)
        ml_runid = run.info.run_id
        print(f"Run id{ml_runid}")
