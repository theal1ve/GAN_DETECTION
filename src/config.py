from pathlib import Path
import torch

SEED = 777

# Пути
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED_80_20 = DATA_PROCESSED / "clean_80_20"
DATA_PROCESSED_90_10 = DATA_PROCESSED / "clean_90_10"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

for _p in [DATA_PROCESSED, MODELS_DIR, FIGURES_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
    

CSV_PATH = DATA_RAW / "train_solution.csv"
IMAGES_DIR = DATA_RAW / "train_images"
TEST_IMAGES_DIR = DATA_RAW / "test_images"

# Девайс
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Гиперпараметры
BATCH_SIZE = 32
IMG_SIZE = 256
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Нормализация 
NORM_MEAN = [0.5194, 0.4280, 0.3847]
NORM_STD = [0.2861, 0.2640, 0.2636]