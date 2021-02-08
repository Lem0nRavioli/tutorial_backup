""" Learning Keras functional API """

import silence_tensorflow.auto
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential, Model
import numpy as np


def compare_seq_vs_api():
    seq_model = Sequential()
    seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
    seq_model.add(layers.Dense(32, activation='relu'))
    seq_model.add(layers.Dense(10, activation='softmax'))

    # ==

    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation='relu')(input_tensor)
    x = layers.Dense(32, activation='relu')(x)
    ouput_tensor = layers.Dense(10, activation='softmax')(x)

    model = Model(input_tensor, ouput_tensor)

    seq_model.summary()
    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    x_train = np.random.random((1000, 64))
    y_train = np.random.random((1000, 10))

    model.fit(x_train, y_train, epochs=10, batch_size=128)
    model.evaluate(x_train, y_train)


def multi_input_toy():
    """ Toy example representing a question answering model
        p262, currently broken toy example, data generated is probably the issue """

    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500

    text_input = Input(shape=(None,), dtype='int32', name='text')
    embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
    encoded_text = layers.LSTM(32)(embedded_text)

    question_input = Input(shape=(None,), dtype='int32', name='question')
    embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)

    concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
    answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

    model = Model([text_input, question_input], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # toy data generation part

    num_samples = 1000
    max_length = 100

    text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))

    answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))  # answers are one hot encoded

    # model.fit([text, question], answers, epochs=10, batch_size=128)  # fitting using a list of inputs
    model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)  # fit with dictionary


def multi_ouput_toy():
    """ p265 toy example taking series of post from social media and attempt to output age, gender, income.
        currently also broken
        compile just fine, probably a sample issue again here, come back with real life data """

    vocabulary_size = 50000
    num_income_groups = 10

    posts_input = Input(shape=(None,), dtype='int32', name='posts')
    embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
    x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)

    age_prediction = layers.Dense(1, name='age')(x)
    income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
    gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)

    model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
    model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
    # ==
    model.compile(optimizer='rmsprop',
                  loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'})

    # loss weight can be an issue, this is how to solve it
    # mse usually range around 3-5, catcross 1 and binarycross 0.1
    model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
                  loss_weights=[.25, 1., 10.])

    model.compile(optimizer='rmsprop',
                  loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'},
                  loss_weights={'age': .25, 'income': 1., 'gender': 10.})

    num_samples = 5000
    max_length = 200
    posts = np.random.randint(1, vocabulary_size, size=(num_samples, max_length))
    age_targets = np.random.randint(0, 100, size=num_samples)
    income_targets = np.random.randint(0, num_income_groups, size=num_samples)
    gender_targets = np.random.randint(0, 1, size=num_samples)

    model.fit(posts, [age_targets, income_targets, gender_targets],
              epochs=10, batch_size=64)
    # ==
    model.fit(posts, {'age': age_targets,
                      'income': income_targets,
                      'gender': gender_targets},
              epochs=10, batch_size=64)


# p267