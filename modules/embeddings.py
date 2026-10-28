"""Раздел 1b. Эмбеддинги: позиционное кодирование, collate, pipeline.

Функции и классы:
    sinusoidal_positional_encoding — фиксированное PE (задание 1.3a).
    LearnablePositionalEmbedding   — обучаемое PE (задание 1.3b).
    collate_fn                     — сборка батча с паддингом (задание 1.4).
    TokenEmbeddingPipeline         — входной блок Transformer (задание 1.5).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ======================================================================
# Задание 1.3a. Синусоидальное позиционное кодирование
# ======================================================================

def sinusoidal_positional_encoding(
    max_len: int,
    d_model: int,
) -> torch.Tensor:
    """Вычисляет синусоидальное позиционное кодирование.

    Формулы:
        PE(p, 2k)     = sin(p / 10000^(2k / d_model))
        PE(p, 2k + 1) = cos(p / 10000^(2k / d_model))

    где *p* — позиция (0, 1, …, max_len − 1), *k* — индекс измерения.

    **Важно:** вычисление должно быть полностью векторизовано средствами
    ``torch``. Циклы по позициям или координатам запрещены.

    Args:
        max_len: Максимальная длина последовательности.
        d_model: Размерность эмбеддинга.

    Returns:
        Тензор формы ``(max_len, d_model)`` с ``dtype=torch.float32``.

    Raises:
        ValueError: Если ``max_len < 1`` или ``d_model < 1``.
    """
    pass


# ======================================================================
# Задание 1.3b. Обучаемое позиционное кодирование
# ======================================================================

class LearnablePositionalEmbedding(nn.Module):
    """Обучаемое позиционное кодирование.

    Хранит матрицу эмбеддингов позиций ``(max_len, d_model)`` как
    обучаемый параметр (``nn.Parameter`` или ``nn.Embedding``).
    В ``forward`` прибавляет позиционные эмбеддинги для позиций
    0, 1, …, seq_len − 1 к входному тензору.

    Args:
        max_len: Максимальная длина последовательности.
        d_model: Размерность эмбеддинга.

    Raises:
        ValueError: Если ``max_len < 1`` или ``d_model < 1``.
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        """Инициализирует обучаемую матрицу позиционных эмбеддингов.

        Не забудьте вызвать ``super().__init__()``.

        Args:
            max_len: Максимальная длина последовательности.
            d_model: Размерность эмбеддинга.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прибавляет позиционные эмбеддинги к входному тензору.

        Args:
            x: Тензор формы ``(batch_size, seq_len, d_model)``.

        Returns:
            Тензор той же формы — ``x + positional_embeddings``.

        Raises:
            ValueError: Если ``seq_len > max_len``.
        """
        pass


# ======================================================================
# Задание 1.4. Collate-функция
# ======================================================================

def collate_fn(
    sequences: list[list[int]],
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Собирает батч из последовательностей разной длины с паддингом.

    Все последовательности паддятся до длины наибольшей в батче.
    Паддинг заполняется значением ``pad_id``.

    Args:
        sequences: Список последовательностей (списков целых чисел)
            разной длины.
        pad_id: Идентификатор паддинг-токена (по умолчанию 0).

    Returns:
        Кортеж ``(input_ids, attention_mask)``:
            - ``input_ids``: ``(batch_size, max_len)``, ``dtype=torch.long``.
            - ``attention_mask``: ``(batch_size, max_len)``, ``dtype=torch.bool``.
              ``True`` для реальных токенов, ``False`` для паддинга.

    Raises:
        ValueError: Если ``sequences`` пуст.
    """
    pass


# ======================================================================
# Задание 1.5. Входной эмбеддинг-пайплайн
# ======================================================================

class TokenEmbeddingPipeline(nn.Module):
    """Входной блок Transformer encoder.

    Объединяет в ``forward`` следующие шаги:
        1. Token-эмбеддинг (``nn.Embedding`` с ``padding_idx``).
        2. Масштабирование на ``sqrt(d_model)`` (стандарт из
           *Attention Is All You Need*).
        3. Позиционное кодирование (синусоидальное или обучаемое).
        4. Layer Normalization (опционально).
        5. Dropout.

    Атрибуты класса:
        _VALID_PE_TYPES: Допустимые значения ``pos_encoding``.
    """

    _VALID_PE_TYPES = ("sinusoidal", "learnable")

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        pos_encoding: str = "sinusoidal",
        dropout: float = 0.1,
        layer_norm: bool = True,
        pad_idx: int = 0,
    ) -> None:
        """Инициализирует все компоненты пайплайна.

        Подсказки:
            - Token-эмбеддинг:
              ``nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)``.
            - Синусоидальное PE: вычислите через
              ``sinusoidal_positional_encoding`` и зарегистрируйте как
              буфер (``register_buffer``, ``persistent=False``) —
              он не обучается, но перемещается с ``.to(device)``.
            - Обучаемое PE: создайте экземпляр
              ``LearnablePositionalEmbedding(max_len, d_model)``.
            - LayerNorm: ``nn.LayerNorm(d_model)`` при ``layer_norm=True``,
              иначе ``nn.Identity()``.
            - Dropout: ``nn.Dropout(p=dropout)``.

        Args:
            vocab_size: Размер словаря.
            d_model: Размерность эмбеддинга.
            max_len: Максимальная длина последовательности.
            pos_encoding: ``"sinusoidal"`` или ``"learnable"``.
            dropout: Вероятность dropout.
            layer_norm: Применять ли LayerNorm после сложения эмбеддингов.
            pad_idx: Индекс паддинг-токена.

        Raises:
            ValueError: Если ``pos_encoding`` не входит в
                ``_VALID_PE_TYPES``.
        """
        pass

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Пропускает входные идентификаторы через эмбеддинг-пайплайн.

        Порядок операций:
            1. Token-эмбеддинг → умножить на ``sqrt(d_model)``.
            2. Прибавить позиционное кодирование (срез до ``seq_len``
               для синусоидального; вызов ``LearnablePositionalEmbedding``
               для обучаемого).
            3. LayerNorm (если включён).
            4. Dropout.

        Args:
            input_ids: Тензор ``(batch_size, seq_len)``, ``torch.long``.

        Returns:
            Тензор ``(batch_size, seq_len, d_model)``.

        Raises:
            ValueError: Если ``seq_len > max_len``.
        """
        pass