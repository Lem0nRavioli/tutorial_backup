import silence_tensorflow.auto
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, shutil
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
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

    return model


model = build_model()
# model.summary()
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                    batch_size=20, class_mode='binary')

validation_generator = valid_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                         batch_size=20, class_mode='binary')

# for data_batch, label_batch in train_generator:
#     print('data batch shape', data_batch.shape)
#     print('data label shape', label_batch.shape)
#     print(label_batch)
#     break

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator, validation_steps=50)
model.save('cats_and_dogs_small_1.h5')

# p160