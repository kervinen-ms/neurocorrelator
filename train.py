from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from pytorch_lightning.callbacks import ModelCheckpoint

from neurocorrelator.data import GeoLocalizationDataModule
from neurocorrelator.train_module import GeoLocalizationModel

with initialize(version_base="1.3", config_path="configs/"):
    cfg = compose(config_name="main")


def train():
    with initialize(version_base="1.3", config_path="configs/"):
        cfg = compose(config_name="main")

    data_module = GeoLocalizationDataModule(
        Path.cwd() / cfg.data.dataset.path,
        batch_size=cfg.train.params.batch_size,
        num_workers=cfg.train.params.n_workers,
    )

    model = GeoLocalizationModel(
        embedding_dim=cfg.model.vgg.embedding_dim,
        lr=cfg.train.params.lr,
        margin=cfg.model.vgg.margin,
    )

    # Настройка логгера
    experiment_name = cfg.logging.mlflow.project
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(uri=cfg.logging.mlflow.tracking_uri)
    mlflow_logger = pl.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow.get_tracking_uri(),
    )

    # Настройка обучения
    trainer = pl.Trainer(
        max_epochs=cfg.train.params.n_epochs,
        accelerator="auto",
        devices="auto",
        logger=mlflow_logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val_recall_1",
                dirpath=Path.cwd() / cfg.logging.mlflow.checkpoint_dir,
                mode="max",
                save_top_k=1,
                filename="geoloc-{epoch:02d}-recall@{val_recall_1:.2f}",
            )
        ],
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data_module)
    torch.save(model.feature_extractor.state_dict(), "model")


if __name__ == "__main__":
    train()
