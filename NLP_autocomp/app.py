from flask import Flask, render_template, request 
import pickle 
import numpy as np 
import keras as kr
from keras.preprocessing.sequence import pad_sequences 
import tensorflow as tf 
import re 

app = Flask(__name__, template_folder='templates') 

model = kr.models.load_model('sentence_completion.h5') 
with open("tokenizer.pkl", 'rb') as file: 
	tokenizer = pickle.load(file) 

@app.route('/') 
def home(): 
	return render_template('index.html') 

# Autocompletations function 
def autoCompletations(text, model): 
	# Tokenization and Text vectorization 
	text_sequences = np.array(tokenizer.texts_to_sequences()) 
	# Pre-padding 
	testing = pad_sequences(text_sequences, maxlen = 53, padding='pre') 
	# Prediction 
	y_pred_test = np.argmax(model.predict(testing,verbose=0)) 
	
	predicted_word = '' 
	for word, index in tokenizer.word_index.items(): 
		if index == y_pred_test: 
			predicted_word = word 
			break
	text += " " + predicted_word + '.'
	return text 
# Generate text function 
def generate_text(text, new_words): 
	for _ in range(new_words): 
		text = autoCompletations(text, model)[:-1] 
	return text 


@app.route('/generate', methods=['GET', 'POST']) 
def generate(): 

	# If a form is submitted 
	if request.method == "POST": 
		
		# Get values through input bars 
		text = request.form.get("Text") 
		no_of_words = request.form.get("NoOfWords") 
	
		# Get prediction from the generate_text function written above 
		generated_text = autoCompletations(text, model) 
		
	else: 
		generated_text = "" 
		
	return render_template("generate.html", output = generated_text) 


# Running the app 
if __name__ == "__main__": 
	app.run(debug=True)
