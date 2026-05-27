from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import IMAGE_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR

def get_generators():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True
    ).flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_gen, val_gen