# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:23:47 2019

@author: Bhavuk
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

#load ascii and convert to lowercase
filename = "pm_modi_speeches_repo/english_speeches_date_place_title_text1.txt"
with open(filename, 'rb') as f:
   raw_text = f.read()
raw_text = raw_text.lower()
raw_text = raw_text.decode('utf-8')

#create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total characters: {}\nTotal Vocab: {}".format(n_chars, n_vocab))

#prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([chars_to_int[char] for char in seq_in])
    dataY.append(chars_to_int[seq_out])
n_patterns = len(dataX)
print('Total patterns: {}'.format(n_patterns))

#reshape X
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X/float(n_vocab)
#one hot encode output
y = np_utils.to_categorical(dataY)

#Building the model
model = Sequential()
model.add(CuDNNLSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#define checkpoints
filepath = "weights-improvement-latest.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#train model
model.load_weights(filepath)
model.fit(X, y, epochs=30, batch_size=256, callbacks=callbacks_list,validation_split=0.3)

#reverse mapping
int_to_char = dict((i, c) for i, c in enumerate(chars))

#pick a random seed

start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print("Seed: ")
print("\" ", ''.join([int_to_char[value] for value in pattern]), "\"")

#generate text 
for i in range(100):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x/float(n_vocab)
#    print(i,x)
    prediction = model.predict(x, verbose=0)
#    print("hi",prediction)
    index = np.argmax(prediction)
    result = int_to_char[index]
#    print(len(prediction),':',index,':',result,end='\n')
    print(result,end='')
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone")

