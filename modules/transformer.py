"""Transformer encoder building blocks (pre-norm architecture).

Implements:
    - TransformerEncoderBlock: single pre-norm block (MHA + FFN + residuals).
    - TransformerEncoder: full encoder (embedding + N blocks + final LayerNorm).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.attention import MultiHeadAttention
from modules.embeddings import TokenEmbeddingPipeline


# ======================================================================
# Задание 3.1. Transformer Encoder Block
# ======================================================================

class TransformerEncoderBlock(nn.Module):
    """Один блок Transformer encoder с **pre-norm** архитектурой.

    В pre-norm архитектуре ``LayerNorm`` применяется **до** каждого
    подблока (attention и FFN), а не после, как в оригинальной статье.
    Это улучшает стабильность обучения глубоких моделей.

    Блок состоит из двух последовательных подблоков, каждый из которых
    использует **pre-norm residual connection** вида
    ``x = x + Dropout(Sublayer(LayerNorm(x)))``:

    1. **Self-attention подблок:** вход нормализуется (``LayerNorm``),
       подаётся в ``MultiHeadAttention``, после чего применяется
       ``Dropout``; результат суммируется с исходным входом (residual).

    2. **Feed-forward подблок:** вход нормализуется (``LayerNorm``),
       проходит через ``Linear → GELU → Dropout → Linear → Dropout``;
       результат суммируется с исходным входом (residual).

    Args:
        d_model: Размерность модели / эмбеддинга.
        num_heads: Количество голов внимания.
        d_ff: Внутренняя размерность feed-forward сети.
        dropout: Вероятность dropout (единая для всех трёх dropout-слоёв).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        """Инициализирует все компоненты блока.

        Подсказки по созданию подмодулей:

        **Self-attention подблок:**
            - ``self.norm1 = nn.LayerNorm(d_model)``
            - ``self.self_attn = MultiHeadAttention(d_model, num_heads)``
              (импортируется из ``modules.attention``)
            - ``self.dropout1 = nn.Dropout(dropout)``

        **Feed-forward подблок:**
            - ``self.norm2 = nn.LayerNorm(d_model)``
            - ``self.linear1 = nn.Linear(d_model, d_ff)``
            - ``self.gelu = nn.GELU()``
            - ``self.dropout2 = nn.Dropout(dropout)``
            - ``self.linear2 = nn.Linear(d_ff, d_model)``
            - ``self.dropout3 = nn.Dropout(dropout)``

        Args:
            d_model: Размерность модели.
            num_heads: Количество голов внимания.
            d_ff: Внутренняя размерность FFN.
            dropout: Вероятность dropout.
        """
        pass

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Применяет один блок Transformer encoder к входному тензору.

        Алгоритм (pre-norm):

        **1. Self-attention подблок:**
            a. Сохраните residual: ``residual = x``.
            b. Нормализуйте: ``x_norm = self.norm1(x)``.
            c. Примените self-attention:
               ``attn_out = self.self_attn(x_norm, x_norm, x_norm, mask=mask)``.
               Обратите внимание: ``query = key = value = x_norm``.
               ``MultiHeadAttention.forward`` возвращает один тензор
               (не кортеж).
            d. Dropout + residual:
               ``x = residual + self.dropout1(attn_out)``.

        **2. Feed-forward подблок:**
            a. Сохраните residual: ``residual = x``.
            b. Нормализуйте: ``x_norm = self.norm2(x)``.
            c. Примените FFN:
               ``Linear → GELU → Dropout → Linear → Dropout``.
            d. Residual: ``x = residual + ffn_out``.

        Args:
            x: Тензор ``(batch_size, seq_len, d_model)``.
            mask: Опциональная маска внимания, broadcastable к
                ``(batch_size, num_heads, seq_len, seq_len)``.
                Передаётся в ``MultiHeadAttention`` без изменений.

        Returns:
            Тензор ``(batch_size, seq_len, d_model)``.
        """
        pass


# ======================================================================
# Задание 3.2. Transformer Encoder
# ======================================================================

class TransformerEncoder(nn.Module):
    """Полный Transformer encoder.

    Архитектура:
        ``TokenEmbeddingPipeline → N × TransformerEncoderBlock → LayerNorm``.

    Входные token-id сначала проходят через эмбеддинг-пайплайн (token
    embedding + positional encoding + dropout + LayerNorm), затем
    последовательно через стек encoder-блоков, и в конце — через
    финальную нормализацию.

    Args:
        vocab_size: Размер словаря токенов.
        d_model: Размерность модели / эмбеддинга.
        num_heads: Количество голов внимания в каждом блоке.
        d_ff: Внутренняя размерность FFN в каждом блоке.
        num_layers: Количество stacked encoder-блоков.
        max_len: Максимальная длина последовательности для PE.
        dropout: Вероятность dropout (передаётся во все подмодули).
        pad_idx: Индекс паддинг-токена (передаётся в embedding).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        """Инициализирует embedding-пайплайн, стек блоков и финальный LayerNorm.

        Подсказки:

        **Входной слой:**
            Создайте ``TokenEmbeddingPipeline`` с параметрами:
            ``vocab_size``, ``d_model``, ``max_len``,
            ``pos_encoding="sinusoidal"``, ``dropout=dropout``,
            ``layer_norm=True``, ``pad_idx=pad_idx``.
            Сохраните как ``self.embedding``.

        **Стек блоков:**
            Создайте ``nn.ModuleList`` из ``num_layers`` экземпляров
            ``TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)``.
            Сохраните как ``self.layers``.

        **Финальная нормализация:**
            ``self.final_norm = nn.LayerNorm(d_model)``.

        Args:
            vocab_size: Размер словаря.
            d_model: Размерность модели.
            num_heads: Количество голов.
            d_ff: Размерность FFN.
            num_layers: Число блоков.
            max_len: Макс. длина последовательности.
            dropout: Dropout probability.
            pad_idx: Индекс паддинг-токена.
        """
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Кодирует батч последовательностей token-id.

        Алгоритм:
            1. **Embedding.** Пропустите ``input_ids`` через
               ``self.embedding`` → ``(B, L, d_model)``.
            2. **Преобразование маски.** Если ``attention_mask`` имеет
               форму ``(B, L)`` (2D), добавьте ось для broadcasting:
               ``attention_mask = attention_mask.unsqueeze(1)``
               → ``(B, 1, L)``. Это позволит маске транслироваться
               по осям голов и запросов в MHA.
               Если ``attention_mask is None``, передавайте ``None``.
            3. **Стек блоков.** Последовательно примените каждый блок
               из ``self.layers`` к ``x``, передавая маску.
            4. **Финальная нормализация.** Примените ``self.final_norm``.

        Args:
            input_ids: ``(batch_size, seq_len)`` — целочисленные token-id.
            attention_mask: ``(batch_size, seq_len)``, ``dtype=torch.bool``.
                ``True`` — реальный токен, ``False`` — паддинг.
                Автоматически расширяется до ``(B, 1, L)`` для broadcasting.
                ``None`` — без маскирования.

        Returns:
            Тензор ``(batch_size, seq_len, d_model)``.
        """
        pass