import silence_tensorflow.auto
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imagehash
import PIL
import os


class CFG():
    threshold = .9
    img_size = 512
    seed = 42


root = 'D:\Progs - D\DataSets\Plant_Pathology'
root_train = os.path.join(root, 'train_images')
root_train_shrinked = os.path.join(root, 'train_shrinked')
root_test = os.path.join(root, 'test_images')
root_csv = os.path.join(root, 'train.csv')

paths = os.listdir(root_train)
df = pd.read_csv(root_csv, index_col='image')


def downgrade_dataset():
    for path in paths:
        image = tf.io.read_file(os.path.join(root_train, path))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [CFG.img_size, CFG.img_size])
        image = tf.cast(image, tf.uint8).numpy()
        plt.imsave(os.path.join(root_train_shrinked, path), image)


hash_functions = [
    imagehash.average_hash,
    imagehash.phash,
    imagehash.dhash,
    imagehash.whash
]

image_ids = []
hashes = []

for path in paths:
    image = PIL.Image.open(os.path.join(root_train_shrinked, path))
    hashes.append(np.array([x(image).hash for x in hash_functions]).reshape(-1,))
    image_ids.append(path)


hashes = np.array(hashes)
image_ids = np.array(image_ids)

duplicate_ids = []

for i in range(len(hashes)):
    similarity = (hashes[i] == hashes).mean(axis=1)
    duplicate_ids.append(list(image_ids[similarity > CFG.threshold]))

duplicates = [frozenset([x] + y) for x, y in zip(image_ids, duplicate_ids)]
duplicates = set([x for x in duplicates if len(x) > 1])

print(f'Found {len(duplicates)} duplicates pairs:')
for row in duplicates:
    print(', '.join(row))


print('Writing duplicates to "duplicates.cvs".')
with open('duplicates.csv', 'w') as file:
    for row in duplicates:
        file.write(','.join(row) + '\n')


def show_dup():
    for row in duplicates:

        figure, axes = plt.subplots(1, len(row), figsize=[5 * len(row), 5])

        for i, image_id in enumerate(row):
            image = plt.imread(image_id)
            axes[i].imshow(image)

            axes[i].set_title(df.loc[image_id, 'labels'])
            axes[i].axis('off')

        plt.show()