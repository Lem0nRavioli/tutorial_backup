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
    strategy = tf.distribute.get_strategy()
    batch_size = 16

    img_size = 128
    folds = 5
    seed = 42
    subfolds = 16
    transform = True  # Data augmentation
    epochs = 5


df = pd.read_csv('train_multilabel.csv', index_col='image')
train_images_names = df.index.values.copy()
train_images = []
labels = []
test_loc = '800113bb65efe69e.jpg'


for filename in train_images_names:
    image = tf.io.read_file(os.path.join(CFG.root_train_shrinked, filename))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [CFG.img_size, CFG.img_size])
    image = np.array(image).astype('float32')

    train_images.append(image)
    labels.append(df.loc[filename].values)

images = np.array(train_images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)

train_datagen = ImageDataGenerator(
    rescale=1. / 255, rotation_range=40, width_shift_range=.2, height_shift_range=.2, shear_range=.2,
    zoom_range=.2, horizontal_flip=True, fill_mode='nearest'
)


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
        tf.keras.layers.Dense(6, activation='softmax')
    ])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=.001), metrics=['acc'])
history = model.fit_generator(train_datagen.flow(images, labels, batch_size=32),
                              epochs=10, steps_per_epoch=int(len(images) / 32))

