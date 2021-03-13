""" data processing csv """

import silence_tensorflow.auto
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import csv
import numpy as np

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
              "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
              "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
              "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
              "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
              "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
              "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
              "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while",
              "who","who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
              "your", "yours", "yourself", "yourselves" ]

path = '../../DataSets/bbc/bbc-text.csv'

sentences = []
labels = []

with open(path, 'r') as csvfile:
    data = csv.reader(csvfile)
    next(data)
    for row in data:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)

# print(len(sentences))
# print(sentences[0])
# print(labels[0])

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
padded = padded/2442.
print(padded[0])
print(padded.shape)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)
# label_seq = [item for sublist in label_seq for item in sublist]
# label_seq = np.array(label_seq)
label_seq = keras.utils.to_categorical(label_seq)
print(label_seq[:10])
print(label_word_index)

model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=padded.shape[1:]))
model.add(keras.layers.Dense(264, activation='relu'))
model.add(keras.layers.Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(padded, label_seq, batch_size=128, epochs=10)