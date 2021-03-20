""" text generation official tensorflow tutorial
 see https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/text_generation.ipynb#scrollTo=yG_n40gFzf9s """

import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

path_to_file = '../../DataSets/text_corpus/shakespeare.txt'

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# print(len(text))
# print(text[:250])


# encode text into unicode bytes (bytes probably unecessary)
example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)

vocab = sorted(set(text))
print(len(vocab))

# convert text into tokens
ids_from_chars = preprocessing.StringLookup(vocabulary=vocab)
ids = ids_from_chars(chars)
print(ids)

# convert back into readable sentence
chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)

chars = chars_from_ids(ids)
print(chars)


def text_froms_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy()


print(text_froms_ids(ids))