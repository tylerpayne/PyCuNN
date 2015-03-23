import numpy as np 
import cudamat as cm 
#from numbapro import vectorize,cuda
import multiprocessing as mp
from sklearn import preprocessing as pp
from timeit import default_timer as timer

def init_weights(n_in,n_out,gpu=True):
	w=None
	if gpu is True:
		w = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size = (n_in,n_out)
		))
	else:
		w = np.random.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size = (n_in,n_out)
		)
	return(w)

def zeros(n_in,n_out,gpu=True):
	w=None
	if gpu is True:
		w = cm.CUDAMatrix(np.zeros((n_in,n_out)))
	else:
		w = np.zeros((n_in,n_out))
	return(w)

def load_data(fname):
	global vocab
	vocab = {}
	global inv_vocab
	inv_vocab = []
	global word_idx
	word_idx = 0
	with open(fname,'r+') as doc:
		f = doc.read()
		sentences = f.split('\n')
		del sentences[-1]
		words = f.split(' ')

	for i in range(len(words)-1):
		if words[i] not in vocab:
			vocab[words[i]] = word_idx
			inv_vocab.append(words[i])
			word_idx += 1
	ds = []
	for seq in sentences:
		seq = seq.split(' ')
		del seq[0]
		del seq[-1]
		sent = []
		for w in seq:
			sent.append(w)
		ds.append(sent)
	return ds

def encode(word):
	x = np.zeros((1,word_idx))
	x[0][vocab[word]] = 1.
	return cm.CUDAMatrix(x)

def decode(arr):
	index = arr.argmax(axis=1)
	return inv_vocab[index]


















