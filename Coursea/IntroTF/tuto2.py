import silence_tensorflow.auto
from tensorflow import keras
import matplotlib.pyplot as plt

class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') > .99:
            print('\n Reached 95% accuracy so cancelling training!')
            self.model.stop_training = True


fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 128/drop03/64 => 88.33 acc test
# x_train = x_train.reshape((x_train.shape[0], 28 * 28)) / 255.
# x_test = x_test.reshape((x_test.shape[0], 28 * 28)) / 255.

# conv network => 88.89 acc test with sgd
# conv network => 92.31 acc test with adam (and multiple training)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


def first_model():
    """ 92.31 acc at top """
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, 5, activation='relu', input_shape=x_train.shape[1:]))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(.3))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train, epochs=20, callbacks=[MyCallback()])
    model.evaluate(x_test, y_test)

    return model


def second_model():
    """ 91.24 acc 1 try """
    model = keras.models.Sequential([
      keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
      keras.layers.MaxPooling2D(2, 2),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train, epochs=10, callbacks=[MyCallback()])
    model.evaluate(x_test, y_test)

    return model


def third_model():
    """ 91.5 acc 1 try with sparse """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train, epochs=10, callbacks=[MyCallback()])
    model.evaluate(x_test, y_test)

    return model


model = third_model()


def show_conv(first=0, second=7, third=26, convnum=1, laynum=4):
    f, axarr = plt.subplots(3, laynum)
    FIRST_IMAGE = first
    SECOND_IMAGE = second
    THIRD_IMAGE = third
    CONVOLUTION_NUMBER = convnum
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    for x in range(0, laynum):
        f1 = activation_model.predict(x_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0, x].grid(False)
        f2 = activation_model.predict(x_test[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1, x].grid(False)
        f3 = activation_model.predict(x_test[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2, x].grid(False)

    plt.show()


show_conv(laynum=2)