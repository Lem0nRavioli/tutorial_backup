""" Using callbacks methods to improve first try of training
    ModelCheckpoint, EarlyStopping,\
    LearningRateScheduler, ReduceLROnPlateau, CSVLogger """

import silence_tensorflow.auto
from tensorflow import keras
import numpy as np


def example_callback_syntax_earlystopping_modelcheckpoint():
    """ Not supposed to be run """
    # random model & data
    x, y = None, None
    x_val, y_val = None, None
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation='softmax', input_shape=(None,)))

    # patience parameter is the amount of epoch without improvement
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=1),
                      keras.callbacks.ModelCheckpoint(filepath='my_model.h5', monitor='val_loss', save_best_only=True)]

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.fit(x, y, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=(x_val, y_val))


def example_callback_syntax_reducelronplateau():
    """" Not supposed to be run """
    x, y, x_val, y_val = None, None, None, None
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, activation='softmax', input_shape=(None,)))

    callback_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=10)]
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    model.fit(x, y, epochs=10, batch_size=32, callbacks=callback_list, validation_data=(x_val, y_val))


def example_writing_new_callback():
    """ Not supposed to be run
        available function to call are : on_epoch_begin/end, on_batch_begin/end, on_train_begin/end """

    class ActivationLogger(keras.callbacks.Callback):
        def set_model(self, model):
            self.model = model
            layer_ouputs = [layer.ouput for layer in model.layers]
            self.activations_model = keras.models.Model(model.input, layer_ouputs)  # return activation all layers

        def on_epoch_end(self, epoch, logs=None):
            if self.validation_data is None:
                raise RuntimeError('Requires validation_data.')

            validation_sample = self.validation_data[0][0:1]
            activations = self.activations_model.predict(validation_sample)
            f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
            np.savez(f, activations)
            f.close()