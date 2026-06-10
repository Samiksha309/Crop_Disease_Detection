import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

MLFLOW_EXPERIMENT = "Crop_Disease_CNN"