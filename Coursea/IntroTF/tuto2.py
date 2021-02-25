import silence_tensorflow.auto
from tensorflow import keras


class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < .15:
            print('\n Reached 90% accuracy so cancelling training!')
            self.model.stop_training = True


fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 128/drop03/64 => 88.33 acc test
# x_train = x_train.reshape((x_train.shape[0], 28 * 28)) / 255.
# x_test = x_test.reshape((x_test.shape[0], 28 * 28)) / 255.

# conv network => 88.89 acc test with sgd
# conv network => 91.11 acc test with adam
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, 5, activation='relu', input_shape=x_train.shape[1:]))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(16, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=128, epochs=80, callbacks=[MyCallback()])

model.evaluate(x_test, y_test)