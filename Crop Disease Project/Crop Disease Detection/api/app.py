from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import json

from src.inference import load_model

app = FastAPI()

model = load_model(version=3)

with open("class_indices.json") as f:
    classes = list(json.load(f).keys())

def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = preprocess(img_bytes)

    pred = model.predict(img)
    return {"prediction": classes[np.argmax(pred)]}