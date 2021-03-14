""" imdb_reviews/subwords8k, can't make it work, apparently depleted, see nlp week 2 of Coursea,
    end of courses column
    https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%203.ipynb#scrollTo=_IoM4VFxWpMR"""

import silence_tensorflow.auto
import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
tokenizer =  info.features['text'].encoder