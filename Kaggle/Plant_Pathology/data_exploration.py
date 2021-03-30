# import silence_tensorflow.auto
# import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as plt

csvfile = pd.read_csv('train.csv')

print(csvfile.labels.unique())

