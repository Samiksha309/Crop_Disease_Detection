import json
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from src.dataset import load_data
from src.model import build_model
from src.tuner import run_tuner
from utils.config import EPOCHS, MLFLOW_EXPERIMENT


mlflow.set_experiment(MLFLOW_EXPERIMENT)

callbacks = [
    EarlyStopping(
        patience=3,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        factor=0.3,
        patience=2
    )
]

def train_best_model():

    train_gen, val_gen = load_data()
    best_hp = run_tuner()

    with mlflow.start_run(run_name="best_tuned_model"):

        model = build_model(
            best_hp,
            len(train_gen.class_indices)
        )

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            callbacks=callbacks
        )

        mlflow.log_params(best_hp.values)

        mlflow.log_metric(
            "val_accuracy",
            history.history['val_accuracy'][-1]
        )

        with open("class_indices.json", "w") as f:
            json.dump(train_gen.class_indices, f)

        mlflow.tensorflow.log_model(
            model,
            "model",
            registered_model_name="CropDiseaseCNN"
        )

        model.save("models/best_model.keras")

if __name__ == "__main__":
    train_best_model()