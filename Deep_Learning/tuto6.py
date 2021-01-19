""" Pretrained models """

import silence_tensorflow.auto
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tuto_utils.util_func import plot_history
import numpy as np
import os


# top is Dense layer with 1000 classes
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# conv_base.summary()

sub_dir = "../DataSets/Cat_Dog/subset"

train_dir = os.path.join(sub_dir, 'train')
validation_dir = os.path.join(sub_dir, 'validation')
test_dir = os.path.join(sub_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
epochs = 30

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))  # shape of conv2d trained last layer
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150),
                                            batch_size=batch_size, class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


def make_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    return model


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validaiton_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = make_model()
history = model.fit(train_features, train_labels, batch_size=batch_size, epochs=epochs,
                    validation_data=(validation_features, validaiton_labels))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plot_history(acc, val_acc, epochs, 'Accuracy')
plot_history(loss, val_loss, epochs, 'Loss')


# 173
