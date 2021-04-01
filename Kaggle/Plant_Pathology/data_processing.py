import silence_tensorflow.auto
import tensorflow as tf
import csv
import numpy as np
from PIL import Image
import os
from collections import Counter

base_path = "D:\Progs - D\DataSets\Plant_Pathology"
base_path_train = os.path.join(base_path, "train_images")
csv_path = os.path.join(base_path, "train.csv")


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


filenames, labels = extract_filenames(csv_path)

print(filenames[:10])
print(labels[:10])
print(np.unique(labels))
print(Counter(labels))

# then iterate over filenames and make a tensor of x*y*channel of pictures, then use datagen.flow


def checkimagesize(paths):
    sizes={}
    for p in paths:
        img = Image.open(os.path.join(base_path_train,p))
        if str(img.size) in sizes:
            sizes[str(img.size)]+=1
        else:
            sizes[str(img.size)]=1
    print(sizes)


# checkimagesize(filenames)
# {'(4000, 2672)': 16485, '(4000, 3000)': 665, '(2592, 1728)': 1027, '(4608, 3456)': 123, '(5184, 3456)': 193,
# '(4032, 3024)': 132, '(3024, 4032)': 3, '(3024, 3024)': 3, '(4000, 2248)': 1}

def get_np_images(paths):
    data = []
    for p in paths:
        img = Image.open(os.path.join(base_path_train, p))
        data.append(np.array(img))
    return data


# data = np.array(get_np_images(filenames))

# need to find a way to flow from directory or squish them during loading in data
# currently too big for memory


