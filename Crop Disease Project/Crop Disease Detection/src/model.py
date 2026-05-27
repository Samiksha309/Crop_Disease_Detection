from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    SeparableConv2D
)
from tensorflow.keras.optimizers import Adam


def build_model(num_classes, lr):

    model = Sequential()

    # Initial lightweight feature extractor
    model.add(
        Conv2D(
            16,
            (3,3),
            activation='relu',
            padding='same',
            input_shape=(224,224,3)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Depthwise Separable Block 1
    model.add(
        SeparableConv2D(
            32,
            (3,3),
            activation='relu',
            padding='same'
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Depthwise Separable Block 2
    model.add(
        SeparableConv2D(
            64,
            (3,3),
            activation='relu',
            padding='same'
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Depthwise Separable Block 3
    model.add(
        SeparableConv2D(
            128,
            (3,3),
            activation='relu',
            padding='same'
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Replace Flatten with GAP (HUGE CPU savings)
    model.add(GlobalAveragePooling2D())

    # Small dense head
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='softmax')) 

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model