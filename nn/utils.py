from numbapro import vectorize, cuda, float32
from numbapro.cudalib import cublas
import numpy as np
from timeit import default_timer as timer
import math
import gc
import pickle
from PyCuNN import *



def init_weights(n,gpu=True):
	w=None
	if gpu is True:
		w = np.array(np.random.uniform(
			low=max(-np.sqrt(6. / (n[0] + n[1])),-1),
			high=min(np.sqrt(6. / (n[0] + n[1])),1),
			size = (n[0],n[1])
		),dtype='float32')
		w = cuda.to_device(w)
	else:
		w = np.random.uniform(
			low=-np.sqrt(6. / (n[0] + n[1])),
			high=np.sqrt(6. / (n[0] + n[1])),
			size = (n[0],n[1])
		)
	return(w)

def zeros(n,gpu=True):
	w=None
	if gpu is True:
		w = np.zeros((n[0],n[1]),dtype='float32')
		w = cuda.to_device(w)
	else:
		w = np.zeros((n[0],n[1]))
	return(w)

def asarray(a):
	x = np.zeros((a.shape),dtype='float32')
	a.copy_to_host(x)
	return x

def load_words_data(fname,gpu=False):
	print('Building Dataset')
	global using_embeddings
	using_embeddings = False
	global vocab
	vocab = {}
	global inv_vocab
	inv_vocab = []
	global word_idx
	word_idx = 0
	global total
	with open(fname,'r+') as doc:
		f = doc.read()
		words = f.split(' ')
		total = len(words)

	ds = []
	for i in range(len(words)-1):
		if words[i] not in vocab:
			vocab[words[i]] = word_idx
			inv_vocab.append(words[i])
			word_idx += 1
	for i in range(len(words)-1):
		ds.append([words[i],words[i+1]])
	print('Dataset is Ready')
	return ds

def load_sentences_data(fname,gpu=False,batch_size=1,use_embeddings=True):
	print('Building Dataset')
	global using_embeddings
	using_embeddings = use_embeddings
	global vocab
	vocab = {}
	global inv_vocab
	inv_vocab = []
	global word_idx
	word_idx = 0
	global total
	ds = []
	with open(fname,'r+') as doc:
		f = doc.read()
		sentences = f.split('\n')
		del sentences[-1]
		words = f.split(' ')
		total = len(words)
	if use_embeddings:
		global vectors
		lookup = pickle.load(open('lookup.pickle','r'))
		vectors = pickle.load(open('vectors.pickle','r'))
		total = len(lookup)
		word_idx = vectors[0].shape[1]
		global d_vectors
		if gpu == True:
			vocab[lookup[0]] = cuda.to_device(vectors[0])
		else:
			vocab[lookup[0]] = vectors[0]
		inv_vocab.append(lookup[0])
		d_vectors = vectors[0]
		for i in range(1,len(lookup)):
			if gpu == True:
				vocab[lookup[i]] = cuda.to_device(vectors[i])
			else:
				vocab[lookup[i]] = vectors[i]
			inv_vocab.append(lookup[i])
			d_vectors = np.concatenate((d_vectors,vectors[i]),axis=0)
		d_vectors = cuda.to_device(d_vectors)
		set_vocab(vocab,inv_vocab,vectors,d_vectors)
		for seq in sentences:
				seq = seq.split(' ')
				del seq[0]
				del seq[-1]
				sent = []
				for w in seq:
					if gpu == True:
						w = cuda.to_device(vocab[w])
					sent.append(w)
				ds.append(sent)
	else:
		with open(fname,'r+') as doc:
			f = doc.read()
			sentences = f.split('\n')
			del sentences[-1]
			words = f.split(' ')
			total = len(words)

		for i in range(len(words)-1):
			if words[i] not in vocab:
				vocab[words[i]] = word_idx
				inv_vocab.append(words[i])
				word_idx += 1

		if batch_size == 1:
			for seq in sentences:
				seq = seq.split(' ')
				del seq[0]
				del seq[-1]
				sent = []
				for w in seq:
					if gpu == True:
						w = encode(w)
					sent.append(w)
				ds.append(sent)
		else:
			count = 0
			for seq in sentences:
				seq = seq.split(' ')
				del seq[0]
				del seq[-1]
				sent = encode(seq[0],gpu=False)
				tar = encode(seq[1],gpu=False)
				for w in xrange(1,len(seq)-1):
					sent = np.concatenate((sent,encode(seq[w],gpu=False)),axis=0)
					tar = np.concatenate((tar,encode(seq[w+1],gpu=False)),axis=0)
				ds.append([cuda.to_device(sent),cuda.to_device(tar)])
				'''if count % batch_size == 0
					ds.append(sent)
					count = 0'''

	print('Dataset is Ready')
	return ds

def encode(word,gpu=True):
	if isinstance(word,basestring):
		if using_embeddings == True:
			return cuda.to_device(vocab[word])
		else:
			x = np.zeros((1,word_idx),dtype='float32')
			x[0][vocab[word]] = 1.
			if gpu == True:
				return cuda.to_device(x)
			else:
				return x
	else:
		return word
	

def decode(arr):
	if not isinstance(arr,basestring):
		if using_embeddings == True:
			return most_similar(arr)
		else:
			index = margmax(arr)
			return inv_vocab[index]
	else:
		return arr

