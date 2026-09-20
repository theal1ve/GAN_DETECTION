"""Датасеты и функции подготовки данных"""


import os
import random
import shutil
import csv
from typing import Tuple, List, Dict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CreateDS_Images_CSV(Dataset):
    """Читает изображения из папки, метки из CSV"""
    def __init__(self, csv_path: str, images_folder: str, transform=None):
        self.df = pd.read_csv(csv_path, header=None)
        self.transform = transform
        self.images_folder = images_folder

    def __getitem__(self, index: int):
        image_name = self.df.iloc[index, 0]
        label = int(self.df.iloc[index, 1])
        image_path = os.path.join(self.images_folder, f"{image_name}.png")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df)


class NoiseClearDataset(Dataset):
    """Возвращает пары (зашумлённое, "чистое") изображения"""
    def __init__(self, images_path: str, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.filenames = [f for f in os.listdir(images_path) if f.endswith(".png")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple:
        filename = self.filenames[index]
        image_path = os.path.join(self.images_path, filename)

        bgr = cv2.imread(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        clean = cv2.medianBlur(rgb, 3)

        noise_image = Image.fromarray(rgb)
        clean_image = Image.fromarray(clean)

        if self.transform:
            noise_image = self.transform(noise_image)
            clean_image = self.transform(clean_image)

        return noise_image, clean_image

import os
import shutil
from pathlib import Path

import pandas as pd

IMG_EXTS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp",
            ".pgm", ".tif", ".tiff", ".webp")


def create_balanced_split(
    csv_path: str,
    images_dir: str,
    out_dir: str,
    train_ratio: float = 0.8,
    seed: int = 777,
    overwrite: bool = False,
) -> None:
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    train_dir = out_dir / "train_dataset"
    test_dir = out_dir / "test_dataset"

    # 1. читаем CSV
    df = pd.read_csv(csv_path, header=None, names=["idx_image", "real/fake"])
    df["idx_image"] = df["idx_image"].astype(str).str.strip()
    df["real/fake"] = df["real/fake"].astype(int)

    # 2. строим индекс stem -> реальное имя файла
    stem_to_file = {}
    for f in os.listdir(images_dir):
        p = Path(f)
        if p.suffix.lower() in IMG_EXTS:
            stem_to_file[p.stem] = f

    # 3. отбираем только существующие
    df["filename"] = df["idx_image"].map(stem_to_file)
    df = df.dropna(subset=["filename"]).reset_index(drop=True)

    print(f"файлов на диске: {len(stem_to_file)}")
    print(f"строк в CSV:     {len(pd.read_csv(csv_path, header=None))}")
    print(f"совпало:         {len(df)}")

    if len(df) == 0:
        raise RuntimeError(
            f"Ни один файл из {csv_path} не найден в {images_dir}"
        )

    for cls in (0, 1):
        if (df["real/fake"] == cls).sum() == 0:
            raise RuntimeError(f"Класс {cls} отсутствует в данных")

    # 4. только теперь чистим/создаём out_dir
    if out_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"{out_dir} уже существует. Передайте overwrite=True."
            )
        shutil.rmtree(out_dir)

    for split in (train_dir, test_dir):
        for cls in ("0", "1"):
            (split / cls).mkdir(parents=True, exist_ok=True)

    # 5. копируем
    for cls in (0, 1):
        cls_df = df[df["real/fake"] == cls].sample(frac=1, random_state=seed)
        n = len(cls_df)
        n_train = max(1, min(int(n * train_ratio), n - 1)) if n > 1 else n

        for f in cls_df.iloc[:n_train]["filename"]:
            shutil.copy(images_dir / f, train_dir / str(cls) / f)
        for f in cls_df.iloc[n_train:]["filename"]:
            shutil.copy(images_dir / f, test_dir / str(cls) / f)

        print(f"Класс {cls}: train={n_train}, test={n - n_train}")

    print(f"Готово: {out_dir}")
    

def remove_salt_and_pepper(image: np.ndarray, threshold: int = 40) -> np.ndarray:
    """Заменяет шумные пиксели (сильно отличающиеся от медианы) на медиану"""
    median = cv2.medianBlur(image, 3).astype(np.int16)
    difference = np.abs(image.astype(np.int16) - median)
    noisy_mask = np.max(difference, axis=2) > threshold

    output = image.copy()
    output[noisy_mask] = median[noisy_mask].astype(image.dtype)
    return output


AUG_SUFFIXES = ("flip_", "affine_", "light_", "all_")


def balance_deepfake_dataset(
    real_path: str,
    deepfake_path: str,
    max_rounds: int = 5,
) -> None:
    """Балансирует классы через аугментации минорного класса"""
    real_count = len([f for f in os.listdir(real_path) if f.endswith(".png")])
    files = [f for f in os.listdir(deepfake_path) if f.endswith(".png")]
    originals = [f for f in files if not f.startswith(AUG_SUFFIXES)]

    flip_t = transforms.RandomHorizontalFlip(p=1.0)
    affine_t = transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))
    light_t = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)
    all_t = transforms.Compose([flip_t, affine_t, light_t])

    augmentations = [
        ("flip_", flip_t),
        ("affine_", affine_t),
        ("light_", light_t),
        ("all_", all_t),
    ]

    for _ in range(max_rounds):
        current = len([f for f in os.listdir(deepfake_path) if f.endswith(".png")])
        if current >= real_count:
            break
        for img_file in originals:
            current = len([f for f in os.listdir(deepfake_path) if f.endswith(".png")])
            if current >= real_count:
                break
            img_path = os.path.join(deepfake_path, img_file)
            img = Image.open(img_path).convert("RGB")
            prefix, aug = random.choice(augmentations)
            new_name = f"{prefix}{random.randint(0, 10**9)}_{img_file}"
            aug(img).save(os.path.join(deepfake_path, new_name))

    current = len([f for f in os.listdir(deepfake_path) if f.endswith(".png")])
    if current > real_count:
        to_delete_count = current - real_count
        augmented = [f for f in os.listdir(deepfake_path)
                     if f.endswith(".png") and f.startswith(AUG_SUFFIXES)]
        to_delete = random.sample(augmented, min(to_delete_count, len(augmented)))
        for f in to_delete:
            os.remove(os.path.join(deepfake_path, f))

    current = len([f for f in os.listdir(deepfake_path) if f.endswith(".png")])
    print(f"Итог: real={real_count}, fake={current}")


def show_dataset_counts(train_path: str, test_path: str) -> None:
    """Печатает количество изображений по классам в train и test"""
    for name, path in [("Train", train_path), ("Test", test_path)]:
        print(f"      {name}      ")
        for cls, cls_name in [("0", "real"), ("1", "fake")]:
            n = len(os.listdir(os.path.join(path, cls)))
            print(f"{cls_name}: {n}")