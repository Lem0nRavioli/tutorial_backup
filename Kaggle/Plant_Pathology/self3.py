import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os


class CFG:
    root = 'D:\Progs - D\DataSets\Plant_Pathology'
    root_train = os.path.join(root, 'train_images')
    root_train_shrinked = os.path.join(root, 'train_shrinked')
    root_test = os.path.join(root, 'test_images')
    root_csv = os.path.join(root, 'train.csv')
    root_duplicate = os.path.join(root, 'duplicates.csv')
    classes = [
        'complex',
        'frog_eye_leaf_spot',
        'powdery_mildew',
        'rust',
        'scab',
        'healthy'
    ]
    batch_size = 128

    img_size = 128
    seed = 42
    transform = True  # Data augmentation
    epochs = 10


df = pd.read_csv('train_multilabel.csv', index_col='image')
train_images_names = df.index.values.copy()
train_path = [os.path.join(CFG.root_train_shrinked, img_name) for img_name in train_images_names]
train_labels = df.values.tolist()

ds_train = tf.data.Dataset.from_tensor_slices((train_path, train_labels))


def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (CFG.img_size, CFG.img_size))
    return image, label


def augment(image, label):
    # data augmentation here
    return image, label


ds_train = ds_train.map(read_image).map(augment).batch(CFG.batch_size).prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(6, activation='sigmoid')
    ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(ds_train, epochs=CFG.epochs,
                    steps_per_epoch=int(len(train_path) / CFG.batch_size))