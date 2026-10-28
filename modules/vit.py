"""Vision Transformer (ViT) for image classification.

Implements:
    - PatchEmbedding: image → sequence of patch embeddings via Conv2d.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.embeddings import LearnablePositionalEmbedding
from modules.transformer import TransformerEncoderBlock


# ======================================================================
# Задание 4.1. Patch Embedding
# ======================================================================

class PatchEmbedding(nn.Module):
    """Преобразует изображение в последовательность патч-эмбеддингов.

    Идея Vision Transformer: вместо того чтобы подавать пиксели в свёрточную
    сеть, мы разрезаем изображение на непересекающиеся квадратные патчи
    размера ``patch_size × patch_size``, проецируем каждый патч в вектор
    размерности ``d_model`` и получаем последовательность, которую можно
    подать в обычный Transformer encoder.

    Реализация через ``nn.Conv2d`` с ``kernel_size = stride = patch_size``
    и ``padding = 0``: свёртка одновременно разрезает изображение на патчи
    и проецирует каждый из них в ``d_model``-мерное пространство.

    Пример::

        patch_emb = PatchEmbedding(in_channels=3, patch_size=16, d_model=192)
        x = torch.randn(2, 3, 224, 224)
        out = patch_emb(x)  # (2, 196, 192), т.к. (224/16)^2 = 196

    Args:
        in_channels: Число каналов входного изображения (3 для RGB).
        patch_size: Размер стороны квадратного патча в пикселях.
        d_model: Размерность эмбеддинга каждого патча.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        d_model: int,
    ) -> None:
        """Инициализирует свёрточную проекцию и сохраняет параметры.

        Подсказки:
            - Сохраните ``self.in_channels``, ``self.patch_size``,
              ``self.d_model`` как атрибуты (они понадобятся в задании 4.2).
            - Создайте ``self.proj = nn.Conv2d(...)`` с параметрами:
              ``in_channels``, ``d_model``,
              ``kernel_size=patch_size``, ``stride=patch_size``,
              ``padding=0``.

        Args:
            in_channels: Число каналов входного изображения.
            patch_size: Размер патча.
            d_model: Размерность эмбеддинга.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Преобразует батч изображений в последовательность патч-эмбеддингов.

        Алгоритм:
            1. Проверьте, что ``H`` и ``W`` делятся на ``patch_size``.
               Если нет — бросьте ``ValueError`` с информативным сообщением.
            2. Примените свёрточную проекцию:
               ``(B, C, H, W) → (B, d_model, H/P, W/P)``.
            3. «Разверните» пространственные измерения в одно:
               ``flatten(2)`` → ``(B, d_model, N)``, где ``N = (H/P)*(W/P)``.
            4. Поменяйте оси местами:
               ``transpose(1, 2)`` → ``(B, N, d_model)``.

        Args:
            x: Тензор ``(batch_size, in_channels, H, W)``.

        Returns:
            Тензор ``(batch_size, num_patches, d_model)``, где
            ``num_patches = (H // patch_size) * (W // patch_size)``.

        Raises:
            ValueError: Если ``H`` или ``W`` не делится на ``patch_size``.
        """
        pass