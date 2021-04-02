from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import silence_tensorflow.auto
import tensorflow as tf
import albumentations
import pandas as pd
import numpy as np
import shutil
import os


class CFG:
    root = 'D:\Progs - D\DataSets\Plant_Pathology'
    root_train = os.path.join(root, 'train_images')
    root_train_shrinked = os.path.join(root, 'train_shrinked')
    root_test = os.path.join(root, 'test_images')
    root_csv = os.path.join(root, 'train.csv')
    root_duplicate = os.path.join(root, 'duplicates.csv')
    classes = [
        'complex',
        'frog_eye_leaf_spot',
        'powdery_mildew',
        'rust',
        'scrab',
        'healthy'
    ]
    strategy = tf.distribute.get_strategy()
    batch_size = 16

    img_size = 512
    folds = 5
    seed = 42
    subfolds = 16
    transform = True  # Data augmentation
    epochs = 5


df = pd.read_csv(CFG.root_csv, index_col='image')
init_len = len(df)

with open(CFG.root_duplicate, 'r') as file:
    duplicates = [x.strip().split(',') for x in file.readlines()]

""" Extract labels from dups, if same label, leave 1 pic, if different labels, remove all"""
for row in duplicates:
    unique_labels = df.loc[row].drop_duplicates().values
    # print(df.loc[row])
    # print(df.loc[row].drop_duplicates())
    # print(unique_labels)
    if len(unique_labels) == 1:
        df = df.drop(row[1:], axis=0)
    else:
        df = df.drop(row, axis=0)


print(f'Dropping {init_len - len(df)} duplicate samples.')