""" Pretrained models """

import silence_tensorflow.auto
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tuto_utils.util_func import plot_history, smooth_curve
import numpy as np
import os


# top is Dense layer with 1000 classes
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False  # Freeze the parameters
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


def make_model(conv=False, lr=2e-5):
    model = models.Sequential()
    if conv:
        model.add(conv_base)
        model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    if not conv:
        model.add(layers.Dropout(.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc'])
    return model


def train_model_input_predicted_convbase():
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

    return model


def train_model_full_pipeline_base_frozen():
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=.2,
                                       height_shift_range=.2, shear_range=.2, zoom_range=.2,
                                       horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                        batch_size=batch_size, class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                                  batch_size=batch_size, class_mode='binary')

    model = make_model(conv=True)

    history = model.fit(train_generator, steps_per_epoch=2000/batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=1000/batch_size)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_history(acc, val_acc, epochs, 'Accuracy')
    plot_history(loss, val_loss, epochs, 'Loss')

    return model


def train_model_full_pipeline_top_unfrozen(dense_train_model):
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=.2,
                                       height_shift_range=.2, shear_range=.2, zoom_range=.2,
                                       horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                        batch_size=batch_size, class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                                  batch_size=batch_size, class_mode='binary')

    test_generator = validation_datagen.flow_from_directory(test_dir, target_size=(150, 150),
                                                            batch_size=batch_size, class_mode='binary')

    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    dense_train_model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5),
                              metrics=['acc'])

    history = model.fit(train_generator, steps_per_epoch=2000 / batch_size, epochs=epochs,
                        validation_data=validation_generator, validation_steps=1000 / batch_size)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']


    plot_history(smooth_curve(acc, factor=.8), smooth_curve(val_acc, factor=.8), epochs, 'Accuracy')
    plot_history(smooth_curve(loss, factor=.8), smooth_curve(val_loss, factor=.8), epochs, 'Loss')

    test_loss, test_acc = model.evaluate(test_generator, steps=1000 / batch_size)
    print('test acc: ', test_acc)

    return model


model = train_model_full_pipeline_base_frozen()
train_model_full_pipeline_top_unfrozen(model)

# p181