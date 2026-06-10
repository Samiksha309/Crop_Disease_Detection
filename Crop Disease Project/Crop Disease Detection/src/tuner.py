import keras_tuner as kt
from src.model import build_model
from src.dataset import load_data

def run_tuner():

    train_gen, val_gen = load_data()

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, len(train_gen.class_indices)),
        objective='val_accuracy',
        max_trials=5,
        overwrite=True,
        directory='tuner_results',
        project_name='crop_disease_tuning'
    )

    tuner.search(
        train_gen,
        validation_data=val_gen,
        epochs=5
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("Best Hyperparameters:")
    print(best_hp.values)

    return best_hp