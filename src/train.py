"""Циклы обучения"""


from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import json
from .config import DEVICE


def run_epoch(model, dataloader, loss_function, optimizer=None) -> Tuple[float, float, float]:
    """Одна эпоха: train (если optimizer) или eval"""
    if optimizer is None:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    total_loss, predicts, labels = 0.0, [], []

    for X, y in tqdm(dataloader, leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = loss_function(logits, y)
        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        predicts.extend(preds)
        labels.extend(y.cpu().numpy())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return (
        total_loss / len(dataloader),
        accuracy_score(labels, predicts),
        f1_score(labels, predicts),
    )


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module = None,
    save_name: str = "best_model.pth",
    device: torch.device = DEVICE) -> dict:
    """
    Обучает модель, сохраняет лучший чекпоинт по val F1.
    Возвращает историю метрик
    """
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    model.to(device)
    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, loss_function, optimizer)
        vl_loss, vl_acc, vl_f1 = run_epoch(model, val_loader, loss_function)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        print(f"train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f}")
        print(f"val:   loss={vl_loss:.4f} acc={vl_acc:.4f} f1={vl_f1:.4f}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            torch.save(model.state_dict(), save_name)
            print(f"Сохранено: {save_name} (F1={best_f1:.4f})")

    hist_path = Path(save_name).with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"История сохранена: {hist_path}")

    return history


def train_denoiser_unet(model, loader, epochs, device=DEVICE, lr=1e-3):
    """Обучение UNet-денойзера на L1 loss (noisy -> clean)"""
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train().to(device)

    for epoch in range(epochs):
        all_loss = 0.0
        for noisy, clean in tqdm(loader, leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs}, loss={all_loss / len(loader):.4f}")
    return model


def train_noise2void(model, loader, epochs, device=DEVICE, lr=1e-4, mask_ratio=0.02):
    """Обучение через Noise2Void: модель предсказывает замаскированные пиксели"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    model.to(device)

    for epoch in range(epochs):
        all_loss = 0.0
        for noisy, _ in tqdm(loader, leave=False):
            noisy = noisy.to(device)
            corrupted, mask = n2v_mask(noisy, device, mask_ratio)
            output = model(corrupted)
            loss = criterion(output[mask], noisy[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs}, loss={all_loss / len(loader):.4f}")
    return model


def n2v_mask(batch: torch.Tensor, device: torch.device, mask_ratio: float = 0.02):
    """Создаёт маску и заменяет пиксели на соседние (для N2V)"""
    B, C, H, W = batch.shape
    mask = (torch.rand(B, 1, H, W, device=device) < mask_ratio).expand(-1, C, -1, -1)
    x = torch.randint(-1, 2, (1,), device=device).item()
    y = torch.randint(-1, 2, (1,), device=device).item()
    neighbor = torch.roll(batch, shifts=(x, y), dims=(2, 3))
    corrupted = batch.clone()
    corrupted[mask] = neighbor[mask]
    return corrupted, mask