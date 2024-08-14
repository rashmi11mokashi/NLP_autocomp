import re 
import numpy as np 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import requests

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow
from keras.preprocessing.text import Tokenizer

# import tensorflow as tf 
# from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.layers import Embedding, LSTM, Dense, Dropout 
from keras.models import Sequential 
from keras.optimizers import Adam 
from keras.utils import to_categorical 
import pickle 
import warnings 
warnings.filterwarnings('ignore')

data = pd.read_csv('Shakespeare_data.csv')
print(data.head)

text = []
for i in data['PlayerLine']:
    text.append(i)
print(text[:5])

def clean_text(text):
    pattern = re.compile('[^a-zA-z0-9\s]')
    text = re.sub(pattern,'',text)

    pattern = re.compile('/d+')
    text = re.sub(pattern,'',text)

    text = text.lower()
    return text 

texts = []
for t in text:
    new_text = clean_text(t)
    texts.append(new_text)

texts[:5]


texts = texts[:10000]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

text_sequences = tokenizer.texts_to_sequences(texts)
print('texts --> ', texts[0])
print('Embedding --> ', text_sequences[0])

max_sequence_len = max([len(x) for x in text_sequences])
text_sequences = pad_sequences(text_sequences, maxlen = max_sequence_len, padding = 'pre')

print('Maximum Sequence Length -->>',max_sequence_len) 
print('Text Sequence -->>\n',text_sequences[0]) 
print('Text Sequence Shape -->>',text_sequences.shape)

X, Y = text_sequences[:,:-1], text_sequences[:,-1]
print('First input ', X[0])
print('First output ', Y[0])

word_index = tokenizer.word_index

total_words = len(word_index) + 1
print('Total number of words : ', total_words)

Y = to_categorical(Y, num_classes=total_words)

print('Input shape --> ', X.shape)
print('Output shape --> ', Y.shape)

model = Sequential(name="LSTM_Model")

# adding embedding
model.add(Embedding(total_words,
                   max_sequence_len-1,
                   input_length=max_sequence_len-1))

# adding a LSTM layer
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))

# adding the final output with activation function of softmax
model.add(Dense(total_words, activation='softmax'))

# printing model summary 
print(model.summary())

# Compiling the model 
model.compile( 
	loss="categorical_crossentropy", 
	optimizer='adam', 
	metrics=['accuracy'] 
) 

# Training the LSTM model 
history = model.fit(X, Y, 
					epochs=50, 
					verbose=1)


def autoCompletations(text, model): 
	# Tokenization and Text vectorization 
	text_sequences = np.array(tokenizer.texts_to_sequences()) 
	# Pre-padding 
	testing = pad_sequences(text_sequences, maxlen = max_sequence_len-1, padding='pre') 
	# Prediction 
	y_pred_test = np.argmax(model.predict(testing,verbose=0)) 
	
	predicted_word = '' 
	for word, index in tokenizer.word_index.items(): 
		if index == y_pred_test: 
			predicted_word = word 
			break
	text += " " + predicted_word + '.'
	return text 
	
complete_sentence = autoCompletations('I have seen this', model) 

