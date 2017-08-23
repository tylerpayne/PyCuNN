import numpy as np
import utils
from utils import *
from PyCuNN import *
from numbapro import cuda
from timeit import default_timer as timer
from scipy.spatial.distance import euclidean as euc
import pickle

class nn(object):
	def __init__(self, layers):
		super(nn, self).__init__()
		self.layers = layers
		self.output = None

		self.w1 = init_weights([self.layers[0],self.layers[1]])

		self.b1 = init_weights([1,self.layers[1]])

		self.w2 = init_weights([self.layers[1],self.layers[2]])

		self.b2 = init_weights([1,self.layers[2]])

		self.h = zeros([1,self.layers[1]])
		self.y = zeros([1,self.layers[2]])
		self.output = zeros([1,self.layers[2]])
		self.delta = zeros([1,self.layers[1]])
		self.gw1 = zeros([self.layers[0],self.layers[1]])
		self.gw2 = zeros([self.layers[1],self.layers[2]])
		self.gb1 = zeros([1,self.layers[1]])
		self.gb2 = zeros([1,self.layers[2]])
		self.gOutput = zeros([1,self.layers[2]])
		self.gInput = zeros([1,self.layers[0]])
		self.input = zeros([1,self.layers[0]])

	def forward(self,x):
		mzero(self.input)
		mzero(self.h)
		mzero(self.y)
		mzero(self.output)

		self.input = mcopy(x)
		fp(x,self.w1,self.b1,self.h)
		mtanh(self.h,self.h)
		fp(self.h,self.w2,self.b2,self.y)
		msoftmax(self.y,self.output)
		return self.output

	def backward(self,t):
		mzero(self.gOutput)
		mzero(self.gInput)
		mzero(self.delta)
		mzero(self.gw2)
		mzero(self.gb2)
		mzero(self.gw1)
		mzero(self.gb1)

		mmsubtract(self.output,t,self.gOutput)
		bp(self.gOutput,self.w2,self.gw2,self.gb2,self.h,self.delta)
		mtanh_deriv(self.delta,self.h,self.delta)
		bp(self.delta,self.w1,self.gw1,self.gb1,self.input,self.gInput)

		update_weights(self.w1,self.gw1,0.01)
		update_weights(self.b1,self.gb1,0.01)
		update_weights(self.w2,self.gw2,0.01)
		update_weights(self.b2,self.gb2,0.01)

		#g = np.zeros(self.w1.shape,dtype='float32')
		#self.w1.copy_to_host(g)
		#print(g)

	def train(self,ds,epochs,batch_size=10):

		for epoch in range(epochs):
			start = timer()
			count = 0.
			correct = 0.
			for i in range(len(ds)/batch_size):
				count += 1.
				x = encode(ds[i*batch_size][0],gpu=False)
				t = encode(ds[i*batch_size][1],gpu=False)
				for b in range(batch_size-1):
					x = np.concatenate((x,encode(ds[i*batch_size + b+1][0],gpu=False)))
					t = np.concatenate((t,encode(ds[i*batch_size + b+1][1],gpu=False)))
				x = cuda.to_device(x)
				t = cuda.to_device(t)
				assert x.shape[1] == self.layers[0]
				assert t.shape[1] == self.layers[2]
				print(x.shape)
				self.forward(x)
				print('output',decode(self.output))
				if decode(self.output) == decode(t):
					correct += 1.
				self.backward(t)
			print("Epoch",epoch,"Time:",timer()-start,'output',decode(self.output), 'Accuracy:',correct/count)
			if correct/count > 0.99:
				break

def test_nn():
	ds = load_words_data('../data/ptb.train.short.txt',gpu=True)
	net = nn([utils.word_idx,500,utils.word_idx])
	net.train(ds,100)
	lookup = []
	vectors = []
	for x in range(utils.word_idx):
		word = utils.inv_vocab[x]
		net.forward(encode(word))
		vectors.append(asarray(net.h))
		lookup.append(word)
	for z in range(10):
		dist = 1000
		idx = 0
		w = vectors[25+z]
		for x in range(utils.word_idx):
			if x is not 25+z:
				if euc(w[0],vectors[x][0]) < dist:
					print('New Best:', lookup[x])
					dist = euc(w[0],vectors[x][0])
					idx = x
		print('Closest to',lookup[25+z],'is',lookup[idx])

	lpath = open('lookup.pickle','w')
	vpath = open('vectors.pickle','w')

	pickle.dump(lookup,lpath)
	pickle.dump(vectors,vpath)
