""" working with json files """

import silence_tensorflow.auto
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

path = '../../DataSets/sarcasm_headlines/Sarcasm_Headlines_Dataset.json'

# with open(path, 'r') as f:
#     datastore = json.load(f)
# got error with previous line of code because json file had multiple dict instead of one
datastore = [json.loads(line) for line in open(path, 'r')]

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])


def show_samples(sample=3):
    for i in range(sample):
        print(sentences[i])
        print(labels[i])
        print(urls[i])


tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)