import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os


model = tf.keras.models.load_model("plant_128x128")

root = 'D:\Progs - D\DataSets\Plant_Pathology'
root_test = os.path.join(root, 'test_images')
train_images_names = os.listdir(root_test)
images = []
labels = []


for filename in train_images_names:
    image = tf.io.read_file(os.path.join(root_test, filename))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = np.array(image).astype('float32') / 255.

    images.append(image)

images = np.array(images)
predict = np.array(model.predict(images))
for row in predict:
    row[np.argmax(row)] = 1
rounder = lambda x: 1 if x >= .5 else 0

vfunc = np.vectorize(rounder)
predict_binary = vfunc(predict)
print(predict_binary)

