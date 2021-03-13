""" embedding beginner + embedding visualization with http://projector.tensorflow.org/
    click on "load" button then import the vector.tsv file and the metadata.tsv file, then click on spherzeize data"""

import silence_tensorflow.auto
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
# print(info)
train_data, test_data = imdb['train'], imdb['test']
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# making the str() conversion cuz if not getting TypeError: a bytes-like object is required, not 'dict' from tokenizer
for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
epochs = 10

tokenizer = Tokenizer(oov_token=oov_tok, num_words=vocab_size)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # keras.layers.Flatten(), # 6ms per epoch, 81.4% val_acc
    keras.layers.GlobalAveragePooling1D(),  # 6ms per epoch, 83% val_acc
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(training_padded, training_labels, epochs=epochs, batch_size=128,
          validation_data=(testing_padded, testing_labels))

#######################################################################################################################
import io

embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
out_v.close()
out_m.close()