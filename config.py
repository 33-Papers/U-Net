import torch

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 572 originally
IMAGE_WIDTH = 512  # 572 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/image/"
TRAIN_MASK_DIR = "data/train/mask/"
VAL_IMG_DIR = "data/test/image/"
VAL_MASK_DIR = "data/test/mask/"
