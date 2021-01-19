""" Data augmentation and generators """

import silence_tensorflow.auto
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil
from tuto_utils.util_func import plot_history
import matplotlib.pyplot as plt


# used https://www.kaggle.com/c/dogs-vs-cats/overview dataset
ds_dir = "../DataSets/Cat_Dog/train"
sub_dir = "../DataSets/Cat_Dog/subset"

train_dir = os.path.join(sub_dir, 'train')
validation_dir = os.path.join(sub_dir, 'validation')
test_dir = os.path.join(sub_dir, 'test')


# generate a subset, call it with either cat or dog
def generate_dog_cat_subset(animal, select=(0, 1000, 1500, 2000)):
    sub_train = os.path.join(train_dir, animal)
    os.mkdir(sub_train)
    sub_val = os.path.join(validation_dir, animal)
    os.mkdir(sub_val)
    sub_test = os.path.join(test_dir, animal)
    os.mkdir(sub_test)

    dirs = [sub_train, sub_val, sub_test]

    for y in range(len(select) - 1):
        fnames = [animal + f'.{i}.jpg' for i in range(select[y], select[y + 1])]
        for fname in fnames:
            src = os.path.join(ds_dir, fname)
            dst = os.path.join(dirs[y], fname)
            shutil.copyfile(src, dst)


def assess_dataset_size():
    print('total training cat images:', len(os.listdir('../DataSets/Cat_Dog/subset/train/cat')))
    print('total training dog images:', len(os.listdir('../DataSets/Cat_Dog/subset/train/dog')))
    print('total validation cat images:', len(os.listdir('../DataSets/Cat_Dog/subset/validation/cat')))
    print('total validation dog images:', len(os.listdir('../DataSets/Cat_Dog/subset/validation/dog')))
    print('total test cat images:', len(os.listdir('../DataSets/Cat_Dog/subset/test/cat')))
    print('total test dog images:', len(os.listdir('../DataSets/Cat_Dog/subset/test/dog')))


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

    return model


def start_training(train_generator, validation_generator, batch_size=32, len_train=2000, len_valid=1000,
                   epochs=30, model_name='cats_and_dogs_small_generic.h5'):
    print(len_train, batch_size)
    history = model.fit_generator(train_generator, steps_per_epoch=int(len_train/batch_size), epochs=epochs,
                                  validation_data=validation_generator, validation_steps=int(len_valid/batch_size))
    model.save(model_name)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_history(acc, val_acc, epochs, metric='Accuracy')
    plot_history(loss, val_loss, epochs, metric='Loss')


def test_data_augmentation():
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=.2, height_shift_range=.2, shear_range=.2,
                                 zoom_range=.2, horizontal_flip=True, fill_mode='nearest')

    fnames = [os.path.join('../DataSets/Cat_Dog/subset/train/cat', fname)
              for fname in os.listdir('../DataSets/Cat_Dog/subset/train/cat')]

    img_path = fnames[3]

    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break

    plt.show()


model = build_model()
# model.summary()
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=.2, height_shift_range=.2,
                                   shear_range=.2, zoom_range=.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                    batch_size=32, class_mode='binary')

validation_generator = valid_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                         batch_size=32, class_mode='binary')

# for data_batch, label_batch in train_generator:
#     print('data batch shape', data_batch.shape)
#     print('data label shape', label_batch.shape)
#     print(label_batch)
#     break


start_training(train_generator, validation_generator, epochs=100, model_name='cats_and_dogs_small_2.h5')

# p 161