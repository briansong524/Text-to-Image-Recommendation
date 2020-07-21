import sys
import os
import pickle
import re

import pandas as pd
import numpy as np

from gensim.corpora import Dictionary
from gensim import corpora, models, similarities

from flask import Flask, render_template, request


sys.path.append(os.getcwd())#os.path.abspath('/home/bsong/'))


app = Flask(__name__)

work_dir = 'C:/Users/songb/Documents/Python Scripts/Text-to-Image-Recommendation-master/'
global dummy, df

class similarity:
	'''
	Wrapper class to keep writing clean. 
	'''
	def __init__(self, dict_, tfidf, lsi, cos_index, ind2img, preprocess_text):
		self.dict_ = dict_
		self.tfidf = tfidf
		self.lsi = lsi        
		self.cos_index = cos_index
		self.ind2img = ind2img
		self.preprocess_text = preprocess_text
		
	def get_simil(self, test_string):
		'''
		convert string to lsi representation and find correlating text
		'''
		test_string = self.preprocess_text(test_string).split()
		conv_str = self.lsi[self.tfidf[self.dict_.doc2bow(test_string)]]
		sims = self.cos_index.__getitem__(conv_str)
		if sims == []:
			print('no matches found')
			return None
		else:
			sims = map(lambda x: tuple([self.ind2img[x[0]], x[1]]), sims)
			sims = f7_tup(sims)
			return sims

def f7_tup(seq):
    '''
    modified f7() to work with tuples specific to this script
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x[0] in seen or seen_add(x[0]))]

def preprocess_text(text):
	'''
	Simple (fast) text preprocessing. 
	1. Split text like 'cleaning/detailing' into 'cleaning detailing'
	2. remove special characters (only alphanumeric string should remain)
	3. Lowercase all the words
	
	This is preprocessing text sentences at a time. Slower, but potentially 
	stronger, text preprocessing methods (lemmatizing, stemming, etc.) can be 
	done through nltk or other packages. The slowness of these preprocessing methods
	arises from the fact that those are done one token at a time, so plan accordingly. 
	'''
	def remove_special_char(text):
		special_char = ['.',',','=','-','_','?','!',';',':',"'",'"','(',')','*','&','^','%','$','#','@','~','`','+','/','\\'] #,'?','!',';',':',"'",'"' temporarily added
		for i in special_char:
			text = text.replace(i,'')
		return text
	text = re.sub('/',' ', text) # sometimes people write like 'cleaning/detailing', so replace / with space
	text = remove_special_char(text)
	text = text.lower()
	  
	return text

def import_things():
	dictionary_ = Dictionary.load(work_dir + 'model_files/gensim_dict.dict')
	cos_index = similarities.Similarity.load(work_dir + 'model_files/cos_index.pkl')

	pickle_files = ['tfidf.pkl','lsi.pkl','ind2img.pkl']
	stored_input = list()
	for i in pickle_files:
		with open(work_dir + 'model_files/' + i,'rb') as pickle_in:
			stored_input.append(pickle.load(pickle_in))

	tfidf, lsi, ind2img = stored_input

	captions_raw = []
	with open(work_dir + 'results_20130124.token', encoding='utf-8') as cap_text:
	    for line in cap_text:
	        captions_raw.append(line)

	captions_raw = list(map(lambda x: x.replace('\n',''), captions_raw))
	tuple_pic_cap = map(lambda x: x.split('\t'), captions_raw)
	tuple_pic_cap = list(map(lambda x: tuple([x[0].split('#')[0], preprocess_text(x[1])]), tuple_pic_cap))
	df = pd.DataFrame(tuple_pic_cap, columns=['img','caption'])

	dummy = similarity(dictionary_, tfidf, lsi, cos_index, ind2img, preprocess_text)

	return dummy, df



def print_results(test_string):
	'''
	Print out the results of the model given a test string. May put this in the similarity class.
	'''
	outputs = dict()
	outputs['text'] = test_string
	results = dummy.get_simil(test_string)

	try:
		for i in range(10):
			one_result = results[i]
			jpgname = one_result[0]
			sim_score = one_result[1]
			caption = df[df.img == jpgname].caption.values[0]
			#caption = '\n'.join(caption)
			outputs['image_' + str(i)] = '/static/' + jpgname #work_dir + 'flickr30k_images/' +
			outputs['caption_' + str(i)] = caption 
	except Exception as err:
		print(err)

	return outputs

@app.route('/generate', methods = ['GET'])
def generate_page():

	# model page
	text = request.args.get('text')
	string_output = print_results(text)
	return render_template('generate_page.html', variable=string_output)

	   


if __name__ == '__main__':
	dummy, df  = import_things()
	app.run(debug=False, port=8999, host='0.0.0.0')