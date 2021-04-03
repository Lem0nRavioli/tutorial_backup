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
        'scab',
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


original_labels = df['labels'].values.copy()
df['labels'] = [x.split(' ') for x in df['labels']]
# print(df.head())
labels = MultiLabelBinarizer(classes=CFG.classes).fit_transform(df['labels'].values)

df = pd.DataFrame(columns=CFG.classes, data=labels, index=df.index)
# df.to_csv('train_multilabel.csv')
# print(df.head())

kfold = StratifiedKFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
fold = np.zeros((len(df),))

for i, (train_index, val_index) in enumerate(kfold.split(df.index, original_labels)):
    fold[val_index] = i

""" Count values in the selected frame, then return the normalized count
    .loc[0] return the "not positive" and .loc[1] the "positive" count """
def value_counts(x):
    return pd.Series.value_counts(x, normalize=True)


df_occurence = pd.DataFrame({
    'origin': df.apply(value_counts).loc[1],
    'fold_0': df[fold == 0].apply(value_counts).loc[1],
    'fold_1': df[fold == 1].apply(value_counts).loc[1],
    'fold_2': df[fold == 2].apply(value_counts).loc[1],
    'fold_3': df[fold == 3].apply(value_counts).loc[1],
    'fold_4': df[fold == 4].apply(value_counts).loc[1],
})

# bar = df_occurence.plot.barh(figsize=[15, 5], colormap='plasma')
# plt.show()

folds = pd.DataFrame({
    'image':df.index,
    'fold' : fold
})

# folds.to_csv('folds.csv', index=False)
# df_occurence.to_csv('df_occurence.csv', index=False)

if CFG.transform:
    transform = albumentations.Compose([
        albumentations.RandomResizedCrop(CFG.img_size, CFG.img_size, scale=(0.9, 1), p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
        albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
        albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.2),
        albumentations.OneOf([
            albumentations.GaussNoise(var_limit=[10, 50]),
            albumentations.GaussianBlur(),
            albumentations.MotionBlur(),
            albumentations.MedianBlur(),
        ], p=0.2),
        albumentations.Resize(CFG.img_size, CFG.img_size),
        albumentations.OneOf([
            albumentations.ImageCompression(),
            albumentations.Downscale(scale_min=0.1, scale_max=0.15),
        ], p=0.2),
        albumentations.IAAPiecewiseAffine(p=0.2),
        albumentations.IAASharpen(p=0.2),
        albumentations.CoarseDropout(max_height=int(CFG.img_size * 0.1), max_width=int(CFG.img_size * 0.1), max_holes=5,
                                     p=0.5),
    ])
else:
    transform = None


def show_me_transform():
    figure, axes = plt.subplots(5, 5, figsize=[15, 15])
    axes = axes.reshape(-1,)

    if transform is None:
        for i in range(len(axes)):
            image = tf.io.read_file(os.path.join(CFG.root_train, df.index[i]))
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [CFG.img_size, CFG.img_size])
            image = tf.cast(image, tf.uint8)

            axes[i].imshow(image.numpy())
            axes[i].axis('off')

    else:
        image = tf.io.read_file(os.path.join(CFG.root_train, df.index[CFG.seed]))
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [CFG.img_size, CFG.img_size])
        image = tf.cast(image, tf.uint8)

        for i in range(len(axes)):
            axes[i].imshow(transform(image=image.numpy())['image'])
            axes[i].axis('off')

    plt.show()


# show_me_transform()