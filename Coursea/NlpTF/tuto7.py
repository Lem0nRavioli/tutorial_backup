""" Trying LSTM / CONV1D / GRU / hypertuning to check results with sarcasm dataset """

import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
from tuto_utils.util_func import plot_acc_loss

path = '../../DataSets/sarcasm_headlines/Sarcasm_Headlines_Dataset.json'
datastore = [json.loads(line) for line in open(path, 'r')]

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])


def show_samples(sample=3):
    for i in range(sample):
        print(sentences[i])
        print(labels[i])


vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000
num_epochs = 50

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

""" MODEL TUNING """

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    # tf.keras.layers.Conv1D(128, 5, activation='relu'),
    # tf.keras.layers.GlobalMaxPooling1D(),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
history = model.fit(training_padded, training_labels,
                    epochs=num_epochs, batch_size=128,
                    validation_data=(testing_padded, testing_labels))

plot_acc_loss(history)

