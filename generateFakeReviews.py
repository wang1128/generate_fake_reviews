#The basic ideas were from the the article "The Unreasonable Effectiveness of Recurrent Neural Networks"
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
#The program aim to generate fake reviews by analyzing the reviews from Yelp Chanllenge 2016 by using RNN.
# keras version updated to 2.0.0

from __future__ import print_function

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential

import string


def create_dataset(window_size):

    text = open('Data/JapaneseR_with_punc.txt').readlines()
    text2 = open('Data/JapaneseR_with_punc.txt').read()
    print('corpus length:', len(text))

    chars = sorted(list(set(text2)))
    print(chars)

    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    step = 1
    sentences = []
    next_chars = []
    #z = 0
    #for i in range(0, len(text) - window_size + 1, step):
    #    sentences.append(text[i: i + window_size])
    #    next_chars.append(text[i + 1:i + 1 + window_size])
    #print('nb sequences:', len(sentences))

    for reviews in text:
        #In the clean data part, the review length > 40
        for i in range(0, len(reviews) - window_size + 1, step):

            sentences.append(reviews[i: i + window_size])
            next_chars.append(reviews[i + 1:i + 1 + window_size])

    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), window_size, len(chars)), dtype=np.bool)  # 40 row, len(chars) col, one-hot model
    y = np.zeros((len(sentences), window_size, len(chars)), dtype=np.bool)  # y is also a sequence , or  a seq of 1 hot vectors

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1


    print(X.shape)

    for i, sentence in enumerate(next_chars):
        for t, char in enumerate(sentence):
            y[i, t, char_indices[char]] = 1

    print(y.shape)

    return len(chars), window_size, X, y, char_indices, indices_char

def create_model(input_dimension,input_length,  epoch_num):
    print('Create the model')
    model = Sequential()
    model.add(LSTM(512, input_shape=(input_length,input_dimension), return_sequences=True)) # change version to 2.0.0 input_dim=input_dimension,
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))  # - original
    model.add(Dropout(0.2))
    model.add(Dense(input_dimension, activation='softmax'))  # len(chars) the results ...

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print(model.summary())
    print('Finish Creating')

    filepath = "fake_review.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs=epoch_num, batch_size=128, callbacks=callbacks_list)


    return model

def generate_fake_review(input_dimension,model,char_indices,indices_char):

    seed_string = "sushi"
    print("seed string -->", seed_string)
    print('The generated text is:')
    # sys.stdout.write(seed_string)

    generateText = seed_string

    for i in range(1000):
        x = np.zeros((1, len(seed_string), input_dimension))
        for t, char in enumerate(seed_string):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        # print(preds)

        next_index = np.argmax(preds[len(seed_string) - 1])
        print(next_index)

        next_char = indices_char[next_index]
        seed_string = seed_string + next_char

        generateText = generateText + next_char

    print(generateText)

    text_file = open("Output.txt", "w")
    text_file.write("GenerateText :  %s" % generateText)
    text_file.close()

if __name__ == '__main__':

    input_dimension, input_length, X, y, char_indices, indices_char = create_dataset(window_size=40)
    model = create_model(input_dimension,input_length, epoch_num= 10)
    generate_fake_review(input_dimension, model, char_indices, indices_char)




