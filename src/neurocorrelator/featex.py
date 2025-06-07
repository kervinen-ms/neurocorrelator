import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """Feature extractor для VGG19 с захватом промежуточных признаков"""

    def __init__(self, use_trained=False):
        super().__init__()
        # Загрузка предобученной VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Используем только первые 16 слове
        self.features = vgg.features[:17]

        # Регистрируем хуки для получения признаков
        self.feature1 = None
        self.feature2 = None
        self.features[2].register_forward_hook(self._save_feature1)
        self.features[16].register_forward_hook(self._save_feature2)

        # Длина сконкатенированных признаков
        self.feature_size = self.features[2].out_channels + self.features[16].out_channels

    def _save_feature1(self, module, input, output):
        self.feature1 = output

    def _save_feature2(self, module, input, output):
        self.feature2 = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.features(x)

        # Интерполяция feature1 до размера feature2
        target_size = (self.feature2.shape[2], self.feature2.shape[3])
        feature1_resized = F.interpolate(self.feature1, size=target_size, mode="bilinear", align_corners=True)

        # Объединение признаков
        return torch.cat((feature1_resized, self.feature2), dim=1)
