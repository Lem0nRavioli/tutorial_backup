""" Basic CNN and Attempt Data Augmentation """

import silence_tensorflow.auto
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


epochs = 20
batch_size = 128
len_train = 2000
len_valid = 1000


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=.2, height_shift_range=.2,
                                   shear_range=.2, zoom_range=.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('../../DataSets/Cat_Dog/subset/train', target_size=(150, 150),
                                                    batch_size=batch_size, class_mode='binary')
valid_generator = valid_datagen.flow_from_directory('../../Datasets/Cat_Dog/subset/validation', target_size=(150, 150),
                                                    batch_size=batch_size, class_mode='binary')


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
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.RMSprop(lr=.001), loss='binary_crossentropy', metrics=['acc'])
model.fit(train_generator, steps_per_epoch=int(len_train/batch_size), epochs=epochs, batch_size=20,
          validation_data=valid_generator, validation_steps=int(len_valid/batch_size))


"""
20 epochs, 128 batch_size, adam, basic cnn + dropout, no data augment: 70.9% valid
20 epochs, 128 batch_size, adam, basic cnn + dropout, data augment: 73.4% valid
20 epochs, 128 batch_size, rmsprop_lr=.001, basic cnn + dropout, data augment: 71.3% valid

"""
