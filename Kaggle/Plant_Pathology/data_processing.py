import silence_tensorflow.auto
import tensorflow as tf
import csv
import numpy as np

train_path = 'train.csv'


def extract_filenames(path, have_labels=True):
    filenames = []
    if have_labels:
        labels = []

        with open(path, 'r') as file:
            data = csv.reader(file)
            next(data)
            for row in data:
                filenames.append(row[0])
                labels.append(row[1])
        return filenames, labels

    with open(path, 'r') as file:
        data = csv.reader(file)
        next(data)
        for row in data:
            filenames.append(row[0])

    return filenames


filenames, labels = extract_filenames(train_path)

print(filenames[:10])
print(labels[:10])
print(np.unique(labels))

# then iterate over filenames and make a tensor of x*y*channel of pictures, then use datagen.flow