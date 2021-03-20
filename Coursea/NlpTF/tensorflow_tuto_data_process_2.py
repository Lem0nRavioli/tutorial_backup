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
vocab = sorted(set(text))

# convert text into tokens
ids_from_chars = preprocessing.StringLookup(vocabulary=vocab)

# convert back into readable sentence
chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1).numpy()


# convert text into separate unicode, then tokenize them
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
# convert ids data into a stream
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text)//(seq_length + 1)

# batch dataset, call the .take(i) to take i batches from datasets
sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
# uncomment this to the the effect of processing
# for seq in sequences.take(5):
#     print(text_from_ids(seq))


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


# map sequences into [x,y] of type [hell, ello] from hello
dataset = sequences.map(split_input_target)
# for input_example, target_example in dataset.take(1):
#     print('input:', text_from_ids(input_example))
#     print('ouput:', text_from_ids(target_example))


""" MODEL PART """

BATCH_SIZE = 64
# tf.data is made to work with infinite data, so we use buffer_size to shuffle a buffer_size part of the data
BUFFER_SIZE = 10000
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# print(dataset)


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=None, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            state = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


model = MyModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=embedding_dim, rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batche_size, sequence_length, vocab_size")
# model.summary()
# to not get model stucked in a loop, we need to take a distribution predictions instead of the highest
print(example_batch_predictions[0])
samples_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
print(samples_indices)
samples_indices = tf.squeeze(samples_indices, axis=-1).numpy()
print(samples_indices)