# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:44:44 2019

@author: Bhavuk
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:23:47 2019

@author: Bhavuk
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM,Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys
from string import punctuation
#load ascii and convert to lowercase
filename = "pm_modi_speeches_repo/english_speeches_date_place_title_text1.txt"
with open(filename, 'rb') as f:
   raw_text = f.read()
raw_text = raw_text.lower()
raw_text = raw_text.decode('utf-8')
rt=''.join([c for c in raw_text if c not in punctuation])
words=rt.split()

length=51
sequences=list()
for i in range(length,len(words)):
  seq=words[i-length:i]
  line=' '.join(seq)
  sequences.append(line)
  
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
tokenizer.fit_on_texts(sequences)
line_sequence=tokenizer.texts_to_sequences(sequences)

vocab_size=len(tokenizer.word_index)+1
line_sequence=np.array(line_sequence)
x,y=line_sequence[:,:-1],line_sequence[:,-1]

from tensorflow.keras.utils import to_categorical
y=to_categorical(y,num_classes=vocab_size)
seq_len=x.shape[1]

model=Sequential()
model.add(Embedding(vocab_size,55,input_length =seq_len))
model.add(CuDNNLSTM(128,return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(vocab_size,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

filepath = "weights3-improvement-latest.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.load_weights(filepath)
model.fit(x,y, batch_size =1000, epochs = 5,callbacks=callbacks_list)

def generate(model,tokenizer,seq_length,org_text,n_words):
  result=list()
  text=org_text
  for _ in range(n_words):
    encoded=tokenizer.texts_to_sequences([text])[0]
    encoded=pad_sequences([encoded],maxlen=seq_length,truncating='pre')
    y=model.predict_classes(encoded,verbose=0)
    out_word=''
    for word,index in tokenizer.word_index.items():
      if index==y:
        out_word=word
        break
    text +=' '+word
    result.append(out_word)
  return ' '.join(result)

from tensorflow.keras.preprocessing.sequence import pad_sequences

text=sequences[311]
print(text,'\n')

generated=generate(model,tokenizer,seq_len,text,50)
print(generated)


