"""Раздел 2. Механизм внимания (Attention).

Функции и классы:
    scaled_dot_product_attention — ядро attention из "Attention Is All You Need"
                                   (задание 2.1).
    MultiHeadAttention           — многоголовое внимание (задание 2.2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Задание 2.1. Scaled Dot-Product Attention
# ======================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Вычисляет scaled dot-product attention.

    Формула из *Attention Is All You Need*:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Алгоритм:
        1. Вычислить логиты: ``scores = Q @ K^T / sqrt(d_k)``.
           Используйте ``torch.matmul`` и ``transpose(-2, -1)`` для
           поддержки произвольного числа ведущих измерений (batch, heads, …).
        2. Если ``mask is not None``, заполнить запрещённые позиции
           (где ``mask == False``) значением ``-inf`` **до** softmax.
           Для этого удобно использовать ``torch.finfo(scores.dtype).min``
           и ``scores.masked_fill(~mask, fill_value)``.
        3. Применить ``softmax`` по последней оси (``dim=-1``).
        4. Вычислить выход: ``output = weights @ V``.

    Args:
        query: Тензор запросов формы ``(..., seq_len_q, d_k)``.
        key:   Тензор ключей формы   ``(..., seq_len_k, d_k)``.
        value: Тензор значений формы ``(..., seq_len_k, d_v)``.
        mask:  Булева маска (``dtype=torch.bool``) формы
               ``(..., seq_len_q, seq_len_k)`` или
               ``(..., 1, seq_len_k)`` (широковещание по оси запросов).
               ``True`` — позиция **разрешена**, ``False`` — **запрещена**.
               ``None`` — без маскирования.

    Returns:
        Кортеж ``(output, weights)``:
            - ``output``:  ``(..., seq_len_q, d_v)`` — взвешенная сумма значений.
            - ``weights``: ``(..., seq_len_q, seq_len_k)`` — веса внимания
              после softmax. На запрещённых позициях веса равны нулю.

    Note:
        Функция должна поддерживать произвольный ведущий батч (например,
        ``(batch, num_heads, seq_len, d_k)`` для multi-head attention).
        Используйте отрицательные индексы осей (``-1``, ``-2``) и
        ``torch.matmul`` вместо фиксированных индексов.
    """
    pass


# ======================================================================
# Задание 2.2. Multi-Head Attention
# ======================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention из *Attention Is All You Need*.

    Реализует многоголовое внимание с четырьмя обучаемыми линейными
    проекциями: ``W_q``, ``W_k``, ``W_v`` (входные) и ``W_o`` (выходная).
    Каждая проекция имеет размер ``(d_model, d_model)``.

    Разделение на головы выполняется через ``reshape`` + ``transpose``
    после единой линейной проекции, что математически эквивалентно
    ``num_heads`` отдельным проекциям ``(d_model → d_k)``, но
    существенно быстрее на GPU.

    Поддерживает как **self-attention** (``query = key = value``),
    так и **cross-attention** (разные последовательности для Q и K/V).

    Args:
        d_model: Размерность модели (должна делиться на ``num_heads``).
        num_heads: Количество голов внимания.

    Raises:
        ValueError: Если ``d_model`` не делится на ``num_heads``.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        """Инициализирует линейные проекции и сохраняет гиперпараметры.

        Подсказки:
            - Проверьте, что ``d_model % num_heads == 0``, иначе бросьте
              ``ValueError``.
            - Сохраните ``self.d_model``, ``self.num_heads`` и
              ``self.d_k = d_model // num_heads``.
            - Создайте четыре ``nn.Linear(d_model, d_model, bias=True)``:
              ``self.w_q``, ``self.w_k``, ``self.w_v``, ``self.w_o``.

        Args:
            d_model: Размерность модели.
            num_heads: Количество голов внимания.
        """
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Вычисляет многоголовое внимание.

        Алгоритм:
            1. **Линейные проекции.** Примените ``w_q``, ``w_k``, ``w_v``
               к ``query``, ``key``, ``value`` соответственно.
               Форма: ``(B, L, d_model) → (B, L, d_model)``.

            2. **Разделение на головы.** Для каждого из Q, K, V:
               ``(B, L, d_model) → (B, L, num_heads, d_k)
               → (B, num_heads, L, d_k)``.
               Используйте ``.view()`` и ``.transpose(1, 2)``.

            3. **Маска.** Если ``mask is not None``, добавьте ось голов
               через ``mask.unsqueeze(1)`` для корректного broadcasting
               по ``(B, 1, ...) → (B, num_heads, ...)``.

            4. **Attention.** Вызовите ``scaled_dot_product_attention``
               из задания 2.1 с подготовленными Q, K, V и маской.

            5. **Сборка голов.** Обратный reshape:
               ``(B, num_heads, L_q, d_k) → (B, L_q, num_heads, d_k)
               → (B, L_q, d_model)``.
               Не забудьте ``.contiguous()`` перед ``.view()``.

            6. **Выходная проекция.** Примените ``w_o``.

        Args:
            query: ``(batch_size, seq_len_q, d_model)``.
            key:   ``(batch_size, seq_len_k, d_model)``.
            value: ``(batch_size, seq_len_k, d_model)``.
            mask:  ``(batch_size, 1, seq_len_k)`` или
                   ``(batch_size, seq_len_q, seq_len_k)``,
                   ``dtype=torch.bool``. ``True`` — разрешено,
                   ``False`` — запрещено. ``None`` — без маски.

        Returns:
            Тензор ``(batch_size, seq_len_q, d_model)``.
        """
        pass