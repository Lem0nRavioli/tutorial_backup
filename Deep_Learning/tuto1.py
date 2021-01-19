""" Basic keras models """

import silence_tensorflow.auto
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten
from tuto_utils.util_func import show_pic

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)  # (60000,) => (60000, 10)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(Conv2D(32, 5, input_shape=(28, 28, 1), activation='relu'))  # conv2D need color channel (28,28,1)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.2))
model.add(Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'test loss: {test_loss}, test acc: {test_acc}')
show_pic(train_images[4].reshape(28, 28))


# p 70

# model = models.Sequential()
# model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
# model.add(layers.Dense(10, activation='softmax'))
#
# # # here same model is builded
# # input_tensor = layers.Input(shape=(784,))
# # x = layers.Dense(32, activation='relu')(input_tensor)
# # output_tensor = layers.Dense(10, activation='softmax')(x)
# #
# # model = models.Model(inputs=input_tensor, outputs=output_tensor)
#
# model.compile(optimizer=optimizers.RMSprop(lr=.001),
#               loss='mse',
#               metrics=['accuracy'])
