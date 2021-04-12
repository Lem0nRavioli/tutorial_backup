import silence_tensorflow.auto
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def model_conv_0(img_shape, outshape):
    model = keras.models.Sequential([
        layers.Input(img_shape),
        layers.Conv2D(16, 3, padding='same'),
        layers.Conv2D(32, 3, padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(outshape, activation='softmax'),
    ])

    return model


