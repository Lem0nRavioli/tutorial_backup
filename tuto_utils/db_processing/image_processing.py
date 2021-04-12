import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

""" https://www.youtube.com/watch?v=q7ZuZ8ZOErE """

TRAIN_DIR = "<<DIRECTORY HERE>>"
X_train = ['Your img path here']
y_train = ['Labels here']
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))

def read_image(image_file, label):
    image = tf.io.read_file(TRAIN_DIR + image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (256, 256))
    return image, label


def augment(image, label):
    # data augmentation here
    return image, label


ds_train = ds_train.map(read_image).map(augment).batch(32)



###########################################################

train = pd.read_csv('<<csv_path>>')
kaggle_path = "../input/hotel-id-2021-fgvc8/train_images/"
test_path = "../input/hotel-id-2021-fgvc8/test_images/"
train['full_filepath'] = kaggle_path + train.chain.astype(str) +"/"+ train.image.astype(str)
train["hotel_id"] = train.hotel_id.astype("str")
X_train, X_val, = train_test_split(train, test_size=0.3,
    stratify = train['hotel_id'], shuffle = True
)

n_classes = X_train.hotel_id.nunique()

BATCH_SIZE = 4
STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
EPOCHS = 15

IMG_HEIGHT = 226
IMG_WIDTH = 226
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = gen.flow_from_dataframe(
    X_train,
    #     directory="../input/hotel-id-2021-fgvc8/train_images",
    x_col="full_filepath",
    y_col="hotel_id",
    weight_col=None,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="training",
    interpolation="nearest",
    validate_filenames=False)

val_gen = gen.flow_from_dataframe(
    X_val,
    #     directory="../input/hotel-id-2021-fgvc8/train_images",
    x_col="full_filepath",
    y_col="hotel_id",
    weight_col=None,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="validation",
    interpolation="nearest",
    validate_filenames=False)