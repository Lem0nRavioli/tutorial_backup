""" Transfer learning with inception """

import silence_tensorflow.auto
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tuto_utils.util_func import plot_acc_loss

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=.2, height_shift_range=.2,
                                   shear_range=.2, zoom_range=.2, horizontal_flip=True, fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('../../DataSets/Cat_Dog/subset/train', batch_size=20,
                                                    class_mode='binary', target_size=(150, 150))
valid_generator = valid_datagen.flow_from_directory('../../DataSets/Cat_Dog/subset/validation', batch_size=20,
                                                    class_mode='binary', target_size=(150, 150))


local_weights_file = '../../Models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

print(last_output)

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
print(pre_trained_model.input)
model.compile(optimizer=RMSprop(lr=.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=20, validation_data=valid_generator, validation_steps=50)

plot_acc_loss(history)


