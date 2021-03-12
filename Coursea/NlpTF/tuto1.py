import silence_tensorflow.auto
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['I love my dog',
             'I love my cat',
             'You love my dog!',
             'Do you think my dog is amazing?']

test_data = ['i really love my dog',
             'my dog loves my manatee']

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)

print(word_index)
print(sequences)
print(padded)

test_seq = tokenizer.texts_to_sequences(test_data)

# new word not in fit_on_texts get lost in conversion
print(test_seq)

print()
print('Adding oov_token parameter to the Tokenizer object')
print('Also adding some parameters to padding for vocabulary purpose')
print()

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

print(word_index)
print(sequences)
print(padded)

test_seq = tokenizer.texts_to_sequences(test_data)

# new word is replaced by the value of oov_token parameter
print(test_seq)