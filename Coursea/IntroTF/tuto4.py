""" Datagen syntax + rollback on data augmentation"""

import silence_tensorflow.auto
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

len_train = 1027
len_valid = 256
epochs = 10
batch_size = 32  # batch size have to be the same for the generator and the step per epoch
# note that you can have different batch_size between training and validation, but still have to calculate the steps
# as int(len_data / batch_size)

# train_datagen = ImageDataGenerator(rescale=1./255)  # vanilla datagen
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=.2, height_shift_range=.2,
                                   shear_range=.2, zoom_range=.2, horizontal_flip=True, fill_mode='nearest')
valid_datagen= ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('../../DataSets/horse-or-human/train', target_size=(300, 300),
                                                    batch_size=batch_size, class_mode='binary')
valid_generator = valid_datagen.flow_from_directory('../../DataSets/horse-or-human/validation', target_size=(300, 300),
                                                    batch_size=batch_size, class_mode='binary')


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, 3, activation='relu', input_shape=(300, 300, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(32, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(.4))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=keras.optimizers.RMSprop(lr=.001), loss='binary_crossentropy', metrics=['acc'])
model.fit(train_generator, steps_per_epoch=int(len_train/batch_size), epochs=epochs,
          validation_data=valid_generator, validation_steps=int(len_valid/batch_size))


""" Result for 10 epochs, bsize 32:
No datagen : 84% 
datagen :  84%
need more epochs, but not time today
"""


def see_conv():
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    import os
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    train_horse_dir = '../../DataSets/horse-or-human/train/horses'
    train_horse_names = os.listdir(train_horse_dir)
    train_human_dir = '../../DataSets/horse-or-human/train/humans'
    train_human_names = os.listdir(train_human_dir)

    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # Let's prepare a random input image from the training set.
    horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
    human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
    img_path = random.choice(horse_img_files + human_img_files)
    img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255.

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers[1:]]

    # Now let's display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            n_features = feature_map.shape[-1]  # number of features in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                # x /= x.std()  # for some reasons this shit divide by 0
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()


# see_conv()