import numpy as np 
import utils
from utils import *
from cuPyNN import *
import lstm
import nn
from timeit import default_timer as timer

class cclstm(object):
	def __init__(self, num_chars, compression_vector_size):
		super(cclstm, self).__init__()
		self.autoencoder = nn.nn([num_chars,1000,250,100,250,1000,num_chars])
		self.charlstm = lstm.lstm([self.autoencoder.layers[3],compression_vector_size,self.autoencoder.layers[3]],softmax=False)
		self.wordlstm = lstm.lstm([compression_vector_size,compression_vector_size + (compression_vector_size/2),compression_vector_size],softmax=False)
		self.sentlstm = lstm.lstm([self.wordlstm.layers[1],self.wordlstm.layers[1] + (self.wordlstm.layers[1]/2),self.wordlstm.layers[1]],softmax=False)
		self.responder = lstm.lstm([self.sentlstm.layers[1],self.sentlstm.layers[1] + (self.se.layers[1]/2),self.sentlstm.layers[1]],softmax=False)
		self.compression_vector_size = compression_vector_size
		self.response_vectors = []
		self.response_lookup = []

	#1
	def train_encoder(self,char_ds,word_ds,sent_ds):
		self.lr = 0.05
		self.charlstm.train(char_ds,200,lr=self.lr)
		self.wordlstm.train(word_ds,200,lr=self.lr)
		self.sentlstm.train(sentlstm,200,lr=self.lr)
	#2
	def train_responder(self,ds):
		self.responder.train(ds,200,lr=0.05)

	#3
	def populate_responses(self,ds):
		for i in ds:
			r = self.forward_prompt(i[0])
			self.response_vectors.append(r)
			self.response_lookup.append([i[1],i[2]])

	def encode_text(self,text):
		sentences = text.split('.')
		for s in sentences:
			words = s.split(' ')
			for w in words:
				chars = list(w)
				for c in chars:
					self.charlstm.forward(encode(c))
				self.wordlstm.forward(self.charlstm.hidden_layer.output)
				self.charlstm.forget()
			self.sentlstm.forward(self.wordlstm.hidden_layer.output)
			self.wordlstm.forget()
		return self.sentlstm.hidden_layer.output

	def forward_prompt(self,prompt):
		p = self.encode_text(prompt)
		r = self.responder.forward(p)
		return r

	def respond_to_prompt(self,prompt):
		r = self.forward_prompt(prompt)
		z = self.best_response(r)
		return z

	def best_response(self,vec):
		z = euclid_dist(vec,self.response_vectors)
		idx = argmin(z)
		return self.response_lookup[idx][0]


	






