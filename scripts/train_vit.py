"""
Fine-tuning предобученной ViT-Tiny на датасете EuroSAT.

Usage:
    python -m scripts.train_vit --epochs 10 --batch-size 64 --lr 1e-3

Артефакты сохраняются в results/:
    - metrics.json
    - loss_curve.png
    - model.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(EUROSAT_CLASSES)

RESULTS_DIR = Path("results")


# ---- вверху файла ----
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False

if _in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# ---------------------------------------------------------------------------
# Адаптер HuggingFace Dataset → PyTorch Dataset
# ---------------------------------------------------------------------------


class HFImageDataset(Dataset):
    """Обёртка вокруг HuggingFace Dataset для использования с PyTorch DataLoader."""

    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, item["label"]


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------


def get_pretrained_vit(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """Загружает предобученную ViT-Tiny из ``timm`` и адаптирует под задачу.

    Алгоритм:
        1. Загрузите модель::

               model = timm.create_model(
                   "vit_tiny_patch16_224", pretrained=True
               )

        2. Замените классификационную голову на новую::

               in_features = model.head.in_features  # 192 для vit_tiny
               model.head = nn.Linear(in_features, num_classes)

        3. Если ``freeze_backbone=True``, заморозьте все параметры
           **кроме** ``model.head`` (установите ``requires_grad=False``
           для всех параметров, чьё имя не начинается с ``"head."``).

        4. Выведите в stdout количество общих и обучаемых параметров::

               Total parameters:     5_537_286
               Trainable parameters: 1_930 (0.035%)

    Args:
        num_classes: Число классов целевой задачи (10 для EuroSAT).
        freeze_backbone: Если ``True``, замораживает все слои кроме головы.

    Returns:
        Готовая к обучению модель ``nn.Module``.
    """
    pass


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def _build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Возвращает (train_transform, eval_transform)."""
    train_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # допустимо для спутниковых снимков
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Загружает датасет EuroSAT и создаёт DataLoader'ы.

    Алгоритм:
        1. Загрузите датасет через HuggingFace::

               hf_dataset = load_dataset("timm/eurosat-rgb")

           Сплиты: ``"train"`` (16 200), ``"validation"`` (5 400),
           ``"test"`` (5 400).

        2. Постройте transforms. Используйте ``_build_transforms(img_size)``
           или создайте их вручную.

           **Train transforms** (в указанном порядке):
               - ``Resize(img_size, interpolation=InterpolationMode.BILINEAR)``
               - ``RandomHorizontalFlip()``
               - ``RandomVerticalFlip()``
               - ``ColorJitter(brightness=0.2, contrast=0.2)``
               - ``ToTensor()``
               - ``Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)``

           **Val/Test transforms:**
               - ``Resize(img_size, interpolation=InterpolationMode.BILINEAR)``
               - ``ToTensor()``
               - ``Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)``

        3. Оберните каждый сплит в ``HFImageDataset`` с соответствующим
           transform.

        4. Создайте ``DataLoader`` для каждого сплита:
           - train: ``shuffle=True``, ``drop_last=True``
           - val/test: ``shuffle=False``
           - ``pin_memory=torch.cuda.is_available()``

    Args:
        batch_size: Размер батча.
        img_size: Размер изображения после Resize (224 для ViT-Tiny).
        num_workers: Число рабочих процессов DataLoader.

    Returns:
        Кортеж ``(train_loader, val_loader, test_loader)``.
    """
    pass


# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Фиксирует seed для воспроизводимости."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Оценка
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Возвращает (avg_loss, accuracy) на loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Собирает все предсказания и истинные метки с loader."""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Цикл обучения
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """Цикл обучения с валидацией после каждой эпохи.

    Алгоритм:
        1. **Seed.** Вызовите ``set_seed(seed)`` для воспроизводимости.

        2. **Подготовка.**
           - Переместите модель на ``device``.
           - ``criterion = nn.CrossEntropyLoss()``.
           - ``optimizer = AdamW`` только по обучаемым параметрам
             (``p.requires_grad == True``), ``lr=lr``, ``weight_decay=0.01``.
           - ``scheduler = CosineAnnealingLR(optimizer, T_max=epochs)``.

        3. **Цикл по эпохам** (от 1 до ``epochs``):
           a. ``model.train()``. Итерируйтесь по ``train_loader``:
              - ``optimizer.zero_grad()``
              - ``logits = model(images)``
              - ``loss = criterion(logits, labels)``
              - ``loss.backward()``
              - ``optimizer.step()``
              - Накапливайте ``running_loss`` (взвешенный по размеру батча).
           b. ``scheduler.step()`` после эпохи.
           c. Вычислите ``val_loss, val_acc`` через ``evaluate(...)``.
           d. Сохраните метрики в ``history``.

        4. Верните словарь ``history`` с ключами
           ``"train_loss"``, ``"val_loss"``, ``"val_acc"``
           (каждый — список ``float`` длины ``epochs``).

    Args:
        model: Модель для обучения.
        train_loader: DataLoader обучающей выборки.
        val_loader: DataLoader валидационной выборки.
        epochs: Число эпох.
        lr: Начальная скорость обучения.
        device: Устройство (``"cuda"`` или ``"cpu"``).
        seed: Seed для воспроизводимости.

    Returns:
        Словарь::

            {
                "train_loss": [float, ...],
                "val_loss":   [float, ...],
                "val_acc":    [float, ...],
            }
    """
    pass


# ---------------------------------------------------------------------------
# Артефакты
# ---------------------------------------------------------------------------


def save_loss_curve(history: dict, path: Path) -> None:
    """Сохраняет график train/val loss и val accuracy."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    ax1.plot(epochs, history["val_loss"], marker="s", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], marker="o", color="green", label="Val accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    """Сохраняет confusion matrix на тестовом наборе."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ViT-Tiny on EuroSAT.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--unfreeze-backbone", action="store_true", default=False,
                        help="Train full model instead of head only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Seed:   {args.seed}")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}")
    print("-" * 60)

    # ----- данные -----
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    # ----- модель -----
    model = get_pretrained_vit(
        num_classes=NUM_CLASSES,
        freeze_backbone=not args.unfreeze_backbone,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ----- обучение -----
    print("-" * 60)
    print("Starting training...")
    start_time = time.time()

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    training_time_sec = time.time() - start_time
    print(f"Training finished in {training_time_sec:.1f} sec.")

    # ----- тест -----
    print("-" * 60)
    print("Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # ----- confusion matrix -----
    print("Building confusion matrix...")
    preds, labels = collect_predictions(model, test_loader, args.device)
    save_confusion_matrix(
        preds, labels, EUROSAT_CLASSES, RESULTS_DIR / "confusion_matrix.png"
    )

    # ----- артефакты -----
    print("Saving artifacts...")

    metrics = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "num_trainable_params": trainable_params,
        "total_params": total_params,
        "training_time_sec": round(training_time_sec, 1),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "img_size": args.img_size,
            "seed": args.seed,
            "freeze_backbone": not args.unfreeze_backbone,
            "model": "vit_tiny_patch16_224",
            "dataset": "timm/eurosat-rgb",
        },
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_loss_curve(history, RESULTS_DIR / "loss_curve.png")

    torch.save(model.state_dict(), RESULTS_DIR / "model.pt")

    print("-" * 60)
    print(f"Done. Artifacts saved to {RESULTS_DIR.resolve()}/")
    print(f"  - metrics.json         (test_acc = {test_acc:.4f})")
    print(f"  - loss_curve.png")
    print(f"  - confusion_matrix.png")
    print(f"  - model.pt             (~{RESULTS_DIR.joinpath('model.pt').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()