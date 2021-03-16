""" Preprocessing json data with padding, tokenizer, extracting embedding with until_function created
    playing with http://projector.tensorflow.org/ again, messing with hyper parameters """

import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
from tuto_utils.util_func import plot_acc_loss, extract_embedding, decode_sentence

path = '../../DataSets/sarcasm_headlines/Sarcasm_Headlines_Dataset.json'

training_size = 20000  # total data is 26709
vocab_size = 3000
embedding_dim = 16
max_length = 20
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
epochs = 20

datastore = [json.loads(line) for line in open(path, 'r')]

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_sentences = np.array(sentences[:training_size])
training_labels = np.array(labels[:training_size])
testing_sentences = np.array(sentences[training_size:])
testing_labels = np.array(labels[training_size:])

tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type,  truncating=trunc_type)

print(decode_sentence(word_index, training_padded[0]))
print(training_sentences[0])
print(labels[0])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    # tf.keras.layers.GlobalAveragePooling1D(),
    # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(training_padded, training_labels, epochs=epochs, batch_size=128,
          validation_data=(testing_padded, testing_labels))

sentence = ["granny starting to fear spiders in the garden might be real",
            "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))

plot_acc_loss(history)
# extract_embedding(model, word_index, vocab_size, file_path='sarcasm_vec_embedding')