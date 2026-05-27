import os
import zipfile
import shutil
import random

DATASET = "emmarex/plantdisease"
SPLIT = 0.8

def download():
    os.system(f"kaggle datasets download -d {DATASET}")

def extract():
    with zipfile.ZipFile("plantdisease.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset")

def split():
    src = "dataset/PlantVillage"
    train_dir = "data/train"
    val_dir = "data/val"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for cls in os.listdir(src):
        cls_path = os.path.join(src, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(train_dir, cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(val_dir, cls, img))

def clean():
    shutil.rmtree("dataset", ignore_errors=True)
    os.remove("plantdisease.zip")

if __name__ == "__main__":
    download()
    extract()
    split()
    clean()