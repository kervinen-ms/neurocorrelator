import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners

from neurocorrelator.featex import VGGFeatureExtractor


class GeoLocalizationModel(pl.LightningModule):
    def __init__(self, embedding_dim: int, lr: float, margin: float):
        super().__init__()
        self.save_hyperparameters()

        # Feature extractor
        self.feature_extractor = VGGFeatureExtractor()

        # Слои для эмбеддингов
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_extractor.feature_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
        )

        # Функция расстояния
        self.distance = distances.CosineSimilarity()

        # Miner для автоматического формирования троек
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        # Loss функция
        self.loss_func = losses.TripletMarginLoss(margin=margin, distance=self.distance, triplets_per_anchor="all")

        # Кэш для функции потерь
        self.train_epoch_loss = 0.0
        self.val_epoch_loss = 0.0

        # Кэш для валидационных эмбеддингов
        self.val_embeddings = []
        self.val_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.embedding.parameters(),  # Оптимизируем только слои эмбеддингов
            lr=self.hparams.lr,
            weight_decay=1e-5,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            # verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_recall_1",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, labels, _ = batch
        embeddings = self(images)
        # Автоматическое формирование троек с помощью miner
        indices_tuple = self.miner(embeddings, labels)
        # Расчет лосса
        loss = self.loss_func(embeddings, labels, indices_tuple)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, _ = batch
        embeddings = self(images)

        # Автоматическое формирование троек с помощью miner
        indices_tuple = self.miner(embeddings, labels)

        # Расчет лосса
        loss = self.loss_func(embeddings, labels, indices_tuple).detach()
        # self.val_epoch_loss += loss

        self.log(
            "val_loss",
            loss,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=images.shape[0],
        )

        # Сохраняем для расчета метрик в конце эпохи
        self.val_embeddings.append(embeddings.detach().cpu())
        self.val_labels.append(labels.detach().cpu())
        return embeddings, labels

    def on_validation_epoch_end(self):
        # Объединяем все эмбеддинги и метки
        all_embeddings = torch.cat(self.val_embeddings)
        all_labels = torch.cat(self.val_labels)

        # Рассчитываем метрики
        recall_1 = self.calculate_recall(all_embeddings, all_labels, k=1)
        recall_5 = self.calculate_recall(all_embeddings, all_labels, k=5)
        recall_10 = self.calculate_recall(all_embeddings, all_labels, k=10)

        # # Логируем лосс
        # self.log("val_loss", self.val_epoch_loss, prog_bar=True)
        # self.val_epoch_loss = 0.0

        # Логируем метрики
        self.log(
            "val_recall_1",
            recall_1,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(self.val_labels),
        )
        self.log("val_recall_5", recall_5, sync_dist=True, batch_size=len(self.val_labels))
        self.log(
            "val_recall_10",
            recall_10,
            sync_dist=True,
            batch_size=len(self.val_labels),
        )

        # Очищаем кэш
        self.val_embeddings.clear()
        self.val_labels.clear()

    # def on_train_epoch_end(self):
    # Логируем лосс
    # self.log("train_loss", self.train_epoch_loss, prog_bar=True, sync_dist=True)
    # self.train_epoch_loss = 0.0

    def calculate_recall(self, embeddings, labels, k=1):
        """Расчет Recall@K с использованием косинусного сходства"""
        # Косинусное сходство
        cos_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        # Маска для исключения сравнения с самим собой
        self_mask = torch.eye(len(labels), dtype=torch.bool)
        cos_sim.masked_fill_(self_mask, -1)

        # Топ-K наиболее похожих
        _, top_k_indices = torch.topk(cos_sim, k + 1, dim=1)
        top_k_indices = top_k_indices[:, 1:]  # Исключаем самого себя

        # Подсчет правильных предсказаний
        correct = 0
        for i in range(len(labels)):
            if labels[i] in labels[top_k_indices[i]]:
                correct += 1

        return correct / len(labels)
