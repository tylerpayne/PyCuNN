import numpy as np 
from timeit import default_timer as timer
import utils
from utils import *

class rnn(object):
	def __init__(self, layers, batch_size=1000):
		super(rnn, self).__init__()
		self.layers = layers
		self.batch_size = batch_size

		self.w1 = init_weights([self.layers[0],self.layers[1]])
		self.b1 = init_weights([1,self.layers[1]])

		self.w2 = init_weights([self.layers[1],self.layers[2]])
		self.b2 = init_weights([1,self.layers[2]])

		self.wr = init_weights([self.layers[1],self.layers[1]])
		self.br = init_weights([1,self.layers[1]])

		self.gw1 = zeros([self.layers[0],self.layers[1]])
		self.gw2 = zeros([self.layers[1],self.layers[2]])
		self.gwr = zeros([self.layers[1],self.layers[1]])
		self.gb1 = zeros([1,self.layers[1]])
		self.gb2 = zeros([1,self.layers[2]])
		self.gbr = zeros([1,self.layers[1]])
		self.gOutput = zeros([self.batch_size,self.layers[2]])
		self.gInput = zeros([self.batch_size,self.layers[2]])
		self.gRecurrent = zeros([self.batch_size,self.layers[1]])
		self.delta = zeros([self.batch_size,self.layers[1]])
		self.updates_tm1 = [self.gw2,self.gb2,self.gw1,self.gb1,self.gwr,self.gbr]

		self.output = zeros([self.batch_size,self.layers[2]])
		self.outputs = []
		self.h = zeros([self.batch_size,self.layers[1]])
		self.prev_h = zeros([self.batch_size,self.layers[1]])
		self.r = zeros([self.batch_size,self.layers[1]])
		self.hs = [zeros([self.batch_size,self.layers[1]])]
		self.inputs=[]
		self.lr = 0.01

	def forward(self,x):
		self.input = mcopy(x)
		fp(x,self.w1,self.b1,self.h)
		fp(self.prev_h,self.wr,self.br,self.r)
		mmadd(self.h,self.r,self.h)
		mtanh(self.h,self.h)
		self.prev_h = mcopy(self.h)
		fp(self.h,self.w2,self.b2,self.output)
		msoftmax(self.output,self.output)
		return self.output

	def bptt(self,t):
		mmsubtract(self.output,t,self.gOutput)
		bp(self.gOutput,self.w2,self.gw2,self.gb2,self.h,self.delta)
			
		mclip(self.delta)
		mtanh_deriv(self.delta,self.h,self.delta)
		mclip(self.delta)
		bp(self.delta,self.w1,self.gw1,self.gb1,self.input,self.gInput)
		#print(np.argmax(asarray(self.inputs[-1])))
		bp(self.delta,self.wr,self.gwr,self.gbr,self.h,self.gRecurrent)
		mclip(self.gbr)
		mclip(self.gwr)
		mclip(self.gw1)
		mclip(self.gb1)
		mclip(self.gw2)
		mclip(self.gb2)

		print(asarray(self.gw2))

	def updateWeights(self):
		#print(asarray(self.gwr))
		update_weights(self.w2,self.gw2,self.lr)
		update_weights(self.b2,self.gb2,self.lr)
		update_weights(self.w1,self.gw1,self.lr)
		update_weights(self.b1,self.gb1,self.lr)
		update_weights(self.wr,self.gwr,self.lr)
		update_weights(self.br,self.gbr,self.lr)
		#print(asarray(self.wr))
		self.forget()

	def train(self,ds,epochs,lr=0.01,decay=0.99):
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		self.lr = lr
		print(len(ds)/self.batch_size)
		for epoch in range(epochs):
			for word in range(len(ds)/self.batch_size):
				batch = encode(ds[word*self.batch_size],gpu=False)
				target = encode(ds[word*self.batch_size + 1],gpu=False)
				for i in range(self.batch_size-1):
					batch = np.concatenate((batch,encode(ds[word*self.batch_size + i+1],gpu=False)))
					target = np.concatenate((target,encode(ds[word*self.batch_size + i+2],gpu=False)))
				batch = cuda.to_device(batch)
				target = cuda.to_device(target)
				print(batch.shape)

				self.forward(batch)
				self.bptt(target)
				self.updateWeights()




	def reset_grads(self):
		mzero(self.gw1)
		mzero(self.gw2)
		mzero(self.gwr)
		mzero(self.gb1)
		mzero(self.gb2)
		mzero(self.gbr)
		mzero(self.gOutput)
		mzero(self.gInput)
		mzero(self.gRecurrent)
		mzero(self.delta)
		self.updates_tm1 = [self.gw2,self.gb2,self.gw1,self.gb1,self.gwr,self.gbr]
		

	def reset_activations(self):
		mzero(self.output)
		self.outputs = []
		mzero(self.h)
		mzero(self.r)
		self.hs = [zeros([self.batch_size,self.layers[1]])]
		self.inputs=[]

	def forget(self):
		self.reset_grads()
		self.reset_activations()

ds = load_words_data('../data/ptb.train.short.txt')

net = rnn([utils.word_idx,1000,utils.word_idx])

net.train(ds,65)
