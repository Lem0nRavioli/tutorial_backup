""" non-binary classifier + cnn + imageDataGen """

import silence_tensorflow.auto
from tensorflow import keras
from tuto_utils.util_func import plot_acc_loss
import os

batch_size = 126
epochs = 20
step_train = int(2520/batch_size)
step_valid = int(327/batch_size)


# ImageDataGenerator classes are in alphabetical order

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=.2,
                                                             height_shift_range=.2, shear_range=.2, zoom_range=.2,
                                                             horizontal_flip=True, fill_mode='nearest')
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('../../DataSets/rps/rps_train', target_size=(150, 150),
                                                    batch_size=batch_size, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory('../../DataSets/rps/rps_test', target_size=(150, 150),
                                                    batch_size=batch_size, class_mode='categorical')


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, 3, activation='relu', input_shape=(150, 150, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(32, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(.5))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=.001), metrics=['acc'])
history = model.fit(train_generator, steps_per_epoch=step_train, epochs=epochs,
                    validation_data=valid_generator, validation_steps=step_valid)

plot_acc_loss(history)


"""
20 epochs, no data augment, plateau 80% valid acc
20 epochs, data augment, 99% valid acc
"""


def try_image(path):
    import numpy as np
    img = keras.preprocessing.image.load_img(path, target_size=(150, 150))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(path)
    print(classes)


try_image('../../DataSets/rps/test_pic/rock3.png')