""" Datagen syntax """

import silence_tensorflow.auto
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('../../DataSets/horse-or-human/train', target_size=(300, 300),
                                                    batch_size=128, class_mode='binary')

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, 3, activation='relu', input_shape=(300, 300, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(32, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.RMSprop(lr=.001), loss='binary_crossentropy', metrics=['acc'])
model.fit(train_generator, epochs=20)
