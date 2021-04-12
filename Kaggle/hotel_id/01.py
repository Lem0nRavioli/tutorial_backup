import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Conv2D
from tensorflow.keras.applications import EfficientNetB0
import cv2
from skimage.transform import resize
import numpy as np
import math


GLOBAL_SEED = 42

np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
path = '../input/hotel-id-2021-fgvc8/'
TRAIN_DIR = path + 'train_images/'
TEST_DIR = path + 'test_images/'
train_df = pd.read_csv('/kaggle/input/hotel-id-2021-fgvc8/train.csv')
train_df['full_filepath'] = TRAIN_DIR + train_df.chain.astype(str) +"/"+ train_df.image.astype(str)

X_train, X_val = train_test_split(train_df, test_size=.2, stratify = train_df['chain'],
                                  random_state = GLOBAL_SEED, shuffle=True)


# n_classes = X_train['hotel_id'].value_counts()  # return id + value count
n_classes = train_df['hotel_id'].nunique()

BATCH_SIZE = 64
STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
EPOCHS = 10

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)


# Based on https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
# https://github.com/keras-team/keras/issues/12847
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/

class HotelBatchSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size,
                 img_size=IMG_SIZE,
                 augment=False):
        """
        `x_set` is list of paths to the images
        `y_set` are the associated classes.

        """

        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """Generate one batch of data"""

        first_id = idx * self.batch_size
        last_id = (idx + 1) * (self.batch_size)

        batch_x = self.x[first_id:last_id]
        batch_y = self.y[first_id:last_id]

        # Xs = np.array([resize(imread(file_name), self.img_size)
        #      for file_name in batch_x])
        #
        # ys = np.array(batch_y)

        output = np.array([
            resize(cv2.imread(file_name), self.img_size)
            for file_name in batch_x]), np.array(batch_y)

        return output


TrainGenerator = HotelBatchSequence(X_train.full_filepath, tf.keras.utils.to_categorical(X_train.hotel_id), BATCH_SIZE)
ValidGenerator = HotelBatchSequence(X_val.full_filepath, tf.keras.utils.to_categorical(X_val.hotel_id), BATCH_SIZE)


def model_0():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


model = model_0()
history = model.fit(TrainGenerator, steps_per_epoch=STEPS_PER_EPOCH, validation_data=ValidGenerator, epochs=10)