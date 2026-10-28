"""Раздел 1a. Токенизация: BPE и SMILES.

Классы:
    BPETokenizer    — Byte Pair Encoding для текста (задание 1.1).
    SmilesTokenizer — regex-based токенизатор для SMILES (задание 1.2).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import ClassVar


# ======================================================================
# Задание 1.1. Byte Pair Encoding
# ======================================================================

class BPETokenizer:
    """Byte Pair Encoding токенизатор.

    Алгоритм BPE итеративно объединяет наиболее частотную пару соседних
    токенов в корпусе, пока размер словаря не достигнет заданного
    ``vocab_size`` или пока не останется частотных пар для слияния.

    При обучении к последнему символу каждого слова добавляется маркер
    конца слова ``END_OF_WORD`` (``</w>``). Это позволяет корректно
    восстанавливать границы слов при декодировании.

    Пример использования::

        tokenizer = BPETokenizer()
        tokenizer.fit(["hello world", "hello there"], vocab_size=20)
        ids = tokenizer.encode("hello world")
        text = tokenizer.decode(ids)

    Атрибуты:
        END_OF_WORD: Маркер конца слова (``"</w>"``).
    """

    END_OF_WORD = "</w>"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Public API — реализуйте эти методы
    # ------------------------------------------------------------------

    def fit(self, corpus: list[str], vocab_size: int) -> None:
        """Обучает BPE-токенизатор на корпусе текстов.

        Алгоритм:
            1. Подсчитать частоты слов в корпусе.
            2. Инициализировать разбиение каждого слова на отдельные символы
               (последний символ получает суффикс ``END_OF_WORD``).
            3. Собрать начальный словарь из всех уникальных символьных токенов.
            4. Повторять, пока ``len(vocab) < vocab_size``:
               a. Подсчитать частоты всех пар соседних токенов.
               b. Если частотных пар нет — завершить досрочно.
               c. Найти наиболее частотную пару и слить её во всех разбиениях.
               d. Добавить слитый токен в словарь (если его там ещё нет).

        Args:
            corpus: Список строк — обучающий корпус.
            vocab_size: Целевой размер словаря.

        Raises:
            ValueError: Если ``vocab_size < 1``.

        Hint:
            Используйте вспомогательные методы ``_build_word_freqs``,
            ``_init_splits``, ``_base_vocab``, ``_compute_pair_freqs``
            и ``_merge_pair``, которые уже реализованы ниже.
        """
        pass

    def encode(self, text: str) -> list[int]:
        """Кодирует произвольную строку в последовательность токен-id.

        Каждое слово токенизируется независимо: разбивается на символы
        (с ``END_OF_WORD`` на конце), затем к нему **последовательно**
        применяются все сохранённые при обучении слияния в том же порядке.

        Args:
            text: Входная строка.

        Returns:
            Список целочисленных идентификаторов токенов.

        Raises:
            RuntimeError: Если токенизатор не обучен (``fit`` не вызван).
            KeyError: Если подслово не найдено в словаре.
        """
        pass

    def decode(self, token_ids: list[int]) -> str:
        """Декодирует последовательность токен-id обратно в строку.

        Токены, оканчивающиеся на ``END_OF_WORD``, сигнализируют о конце
        слова: их содержимое (без маркера) конкатенируется с предыдущими
        токенами текущего слова, после чего слово добавляется в результат.
        Слова разделяются пробелами.

        Args:
            token_ids: Список идентификаторов токенов.

        Returns:
            Восстановленная строка. Для текстов из обучающего корпуса
            ``decode(encode(text)) == text`` с точностью до нормализации
            пробелов.

        Raises:
            RuntimeError: Если токенизатор не обучен.
            KeyError: Если идентификатор не найден в словаре.
        """
        pass

    @property
    def vocab(self) -> dict[str, int]:
        """Возвращает копию текущего словаря ``{токен: id}``."""
        pass

    # ------------------------------------------------------------------
    # Private helpers — уже реализованы, используйте их в методах выше
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Бросает ``RuntimeError``, если токенизатор не обучен."""
        if not self._vocab:
            raise RuntimeError("Tokenizer is not fitted. Call fit() first.")

    @staticmethod
    def _build_word_freqs(corpus: list[str]) -> dict[str, int]:
        """Подсчитывает частоты слов в корпусе.

        Returns:
            Словарь ``{слово: частота}``.
        """
        freq: Counter[str] = Counter()
        for line in corpus:
            for word in line.strip().split():
                freq[word] += 1
        return dict(freq)

    def _init_splits(
        self, word_freqs: dict[str, int]
    ) -> dict[str, list[str]]:
        """Инициализирует разбиение каждого слова на символы + маркер конца.

        Returns:
            Словарь ``{слово: [символ_1, ..., символ_n + END_OF_WORD]}``.
        """
        splits: dict[str, list[str]] = {}
        for word in word_freqs:
            chars = list(word)
            chars[-1] = chars[-1] + self.END_OF_WORD
            splits[word] = chars
        return splits

    @staticmethod
    def _base_vocab(splits: dict[str, list[str]]) -> dict[str, int]:
        """Собирает начальный словарь из всех уникальных символьных токенов.

        Returns:
            Словарь ``{токен: id}``.
        """
        vocab: dict[str, int] = {}
        for parts in splits.values():
            for part in parts:
                if part not in vocab:
                    vocab[part] = len(vocab)
        return vocab

    @staticmethod
    def _compute_pair_freqs(
        splits: dict[str, list[str]], word_freqs: dict[str, int]
    ) -> dict[tuple[str, str], int]:
        """Считает частоту каждой пары соседних токенов во всех разбиениях.

        Returns:
            Словарь ``{(токен_a, токен_b): суммарная_частота}``.
        """
        pair_freqs: Counter[tuple[str, str]] = Counter()
        for word, parts in splits.items():
            freq = word_freqs[word]
            for i in range(len(parts) - 1):
                pair_freqs[(parts[i], parts[i + 1])] += freq
        return dict(pair_freqs)

    @staticmethod
    def _merge_pair(
        pair: tuple[str, str], splits: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Применяет слияние *pair* ко всем разбиениям.

        Returns:
            Новые разбиения с применённым слиянием.
        """
        a, b = pair
        merged = a + b
        new_splits: dict[str, list[str]] = {}
        for word, parts in splits.items():
            new_parts: list[str] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            new_splits[word] = new_parts
        return new_splits

    def _tokenize_word(self, word: str) -> list[str]:
        """Токенизирует одно слово, применяя сохранённые слияния по порядку.

        Returns:
            Список подслов.
        """
        chars = list(word)
        chars[-1] = chars[-1] + self.END_OF_WORD

        for a, b in self._merges:
            merged = a + b
            new_chars: list[str] = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars


# ======================================================================
# Задание 1.2. SMILES-токенизатор
# ======================================================================

class SmilesTokenizer:
    """Regex-based токенизатор для молекулярных формул в формате SMILES.

    Правила токенизации (в порядке убывания приоритета):
        1. Любая подстрока в квадратных скобках — один токен:
           ``[nH]``, ``[C@@H]``, ``[O-]``.
        2. Двухбуквенные обозначения элементов ``Br``, ``Cl`` — один токен.
        3. Остальные символы (буквы, цифры, скобки, ``=``, ``#``, ``/``,
           ``\\``, ``+``, ``-``, ``.``, ``@``, ``%``) — отдельные токены.

    Словарь строится по корпусу SMILES-строк. Первые четыре идентификатора
    (0–3) зарезервированы за специальными токенами ``SPECIAL_TOKENS``.
    Остальные токены упорядочены лексикографически.

    Атрибуты:
        SPECIAL_TOKENS: Список специальных токенов в порядке их id:
            ``["<pad>", "<unk>", "<bos>", "<eos>"]``.
    """

    SPECIAL_TOKENS: ClassVar[list[str]] = ["<pad>", "<unk>", "<bos>", "<eos>"]

    # Regex для токенизации SMILES. Порядок альтернатив гарантирует приоритет.
    _SMILES_REGEX = re.compile(
        r"(\[[^\]]+\]|Br|Cl|[A-Za-z0-9@#%=\\/\+\-\.\(\)])"
    )

    def __init__(self) -> None:
        self._token_to_idx: dict[str, int] = {}
        self._idx_to_token: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API — реализуйте эти методы
    # ------------------------------------------------------------------

    def tokenize(self, smiles: str) -> list[str]:
        """Разбивает SMILES-строку на список токенов согласно правилам.

        Используйте класс-атрибут ``_SMILES_REGEX`` для разбиения.

        Args:
            smiles: SMILES-строка молекулы.

        Returns:
            Список строковых токенов.

        Example::

            >>> tok = SmilesTokenizer()
            >>> tok.tokenize("CC(=O)[O-]")
            ['C', 'C', '(', '=', 'O', ')', '[O-]']
        """
        pass

    def fit(self, smiles_list: list[str]) -> None:
        """Строит словарь по корпусу SMILES-строк.

        Специальные токены из ``SPECIAL_TOKENS`` получают id 0–3
        в указанном порядке. Остальные уникальные токены из корпуса
        добавляются в **лексикографическом** порядке.

        Не забудьте заполнить как ``_token_to_idx``, так и
        ``_idx_to_token``.

        Args:
            smiles_list: Список SMILES-строк для построения словаря.
        """
        pass

    def encode(self, smiles: str, add_special: bool = True) -> list[int]:
        """Кодирует SMILES-строку в список идентификаторов.

        Args:
            smiles: Входная SMILES-строка.
            add_special: Если ``True``, оборачивает результат в
                ``<bos> ... <eos>``.

        Returns:
            Список целочисленных идентификаторов. Неизвестные токены
            заменяются на id ``<unk>``.

        Raises:
            RuntimeError: Если словарь не построен (``fit`` не вызван).
        """
        pass

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        """Декодирует список идентификаторов обратно в SMILES-строку.

        Args:
            token_ids: Список идентификаторов.
            skip_special: Если ``True``, специальные токены
                (``<pad>``, ``<unk>``, ``<bos>``, ``<eos>``) удаляются
                из вывода.

        Returns:
            Восстановленная SMILES-строка (конкатенация токенов).

        Raises:
            RuntimeError: Если словарь не построен.
        """
        pass

    @property
    def vocab(self) -> dict[str, int]:
        """Возвращает копию словаря ``{токен: id}``."""
        pass

    @property
    def vocab_size(self) -> int:
        """Возвращает размер словаря (включая специальные токены)."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Бросает ``RuntimeError``, если словарь не построен."""
        if not self._token_to_idx:
            raise RuntimeError("Tokenizer is not fitted. Call fit() first.")