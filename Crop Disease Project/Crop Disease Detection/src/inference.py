import mlflow.pyfunc
import numpy as np

def load_model(version=3):
    return mlflow.pyfunc.load_model(f"models:/CropDiseaseCNN/{version}")

def predict(model, img_array):
    preds = model.predict(img_array)
    return np.argmax(preds)