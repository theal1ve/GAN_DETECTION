"""
Вспомогательные функции: seed, submission, метрики, визуализация
"""


import os
import csv
import random
from typing import Tuple, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay

from .config import DEVICE, NORM_MEAN, NORM_STD, IMG_SIZE


def set_seed(seed: int = 777) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Прогоняет модель по loader и возвращает (y_true, y_pred)"""
    model.eval()
    model.to(DEVICE)
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def predict_proba_loader(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает вероятности класса 1 (fake) для ROC-AUC"""
    model.eval()
    model.to(DEVICE)
    all_probs, all_labels = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        probs = torch.softmax(model(X), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_probs)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)


def evaluate_model(model: nn.Module, loader: DataLoader, verbose: bool = True) -> dict:
    """Полная оценка: метрики + матрица ошибок.

    Всегда возвращает dict с одинаковым набором ключей:
    accuracy, precision, recall, f1, roc_auc.
    Если в y_true только один класс — roc_auc = nan,
    а classification_report пропускается.
    """
    y_true, y_pred = predict_loader(model, loader)
    _, y_proba = predict_proba_loader(model, loader)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    # roc_auc_score не определён, если в y_true один класс
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y_true, y_proba)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc) if not np.isnan(roc_auc) else float("nan"),
    }

    if verbose:
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1       : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print()

        if len(np.unique(y_true)) < 2:
            print("В y_true только один класс — classification_report пропущен")
        else:
            print(classification_report(y_true, y_pred, target_names=["real", "fake"]))

    return metrics


def plot_confusion_matrix(model, loader, save_path: str = None) -> np.ndarray:
    """Строит и сохраняет матрицу ошибок"""
    y_true, y_pred = predict_loader(model, loader)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["real", "fake"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return cm


def result_csv(model: nn.Module, test_path: str, csv_name: str = "result.csv") -> None:
    """Генерирует submission.csv по изображениям из test_path."""
    from pathlib import Path
    from .config import IMG_SIZE

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    model.eval().to(DEVICE)
    predicts = []

    files = sorted(Path(test_path).iterdir(), key=lambda p: int(p.stem))
    for image_path in tqdm(files, desc="Predict"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = Image.open(image_path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor)
            label = int(logit.argmax(dim=1).item())
        predicts.append([int(image_path.stem), label])

    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target_feature"])
        writer.writerows(sorted(predicts))

    print(f"Submission сохранён: {csv_name}")
    

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Конвертирует тензор [3, H, W] в PIL"""
    tensor = tensor.detach().cpu().clamp(0, 1)
    tensor = (tensor * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(tensor)


def plot_history(history_path: str, save_path: str = None) -> None:
    """Строит графики loss и F1 по эпохам из сохранённой истории"""
    import json
    from pathlib import Path

    history = json.loads(Path(history_path).read_text())
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], label="train")
    axes[1].plot(epochs, history["val_f1"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1-score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()