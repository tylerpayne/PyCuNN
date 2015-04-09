import numpy as np 
from timeit import default_timer as timer
import utils
from utils import *
from scipy.spatial.distance import euclidean as euc
import pickle

class rnn(object):
	def __init__(self, layers):
		super(rnn, self).__init__()
		self.layers = layers

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
		self.gOutput = zeros([1,self.layers[2]])
		self.gInput = zeros([1,self.layers[2]])
		self.gRecurrent = zeros([1,self.layers[1]])
		self.delta = zeros([1,self.layers[1]])
		self.updates_tm1 = [self.gw2,self.gb2,self.gw1,self.gb1,self.gwr,self.gbr]

		self.output = zeros([1,self.layers[2]])
		self.outputs = []
		self.h = zeros([1,self.layers[1]])
		self.r = zeros([1,self.layers[1]])
		self.hs = [zeros([1,self.layers[1]])]
		self.inputs=[]
		self.lr = 0.01

	def forward(self,x):
		assert x.shape[1] == self.layers[0] 
		fp(x,self.w1,self.b1,self.h)
		fp(self.hs[-1],self.wr,self.br,self.r)
		mmadd(self.h,self.r,self.h)
		mtanh(self.h,self.h)
		fp(self.h,self.w2,self.b2,self.output)
		mtanh(self.output,self.output)
		self.inputs.append(mcopy(x))
		self.hs.append(mcopy(self.h))
		self.outputs.append(mcopy(self.output))
		return self.output

	def bptt(self,t):
		for q in range(len(t)-2):
			assert t[-1].shape[1] == self.layers[2] 
			mmsubtract(t[-1],self.outputs[-1],self.gOutput)
			msmult(self.gOutput,-1.,self.gOutput)
			#mmsubtract(self.outputs[-1],t[-1],self.gOutput)
			#print('gOuptut',np.sum(asarray(self.gOutput),axis=1))
			bp(self.gOutput,self.w2,self.gw2,self.gb2,self.hs[-1],self.delta)

			mmadd(self.delta,self.gRecurrent,self.delta)
			#print('delta2',asarray(self.delta))
			mtanh_deriv(self.delta,self.hs[-1],self.delta)
			#print('delta2',asarray(self.delta))
			#mclip(self.delta)
			#print(asarray(self.delta))
			#print(asarray(self.hs[-2]))
			bp(self.delta,self.wr,self.gwr,self.gbr,self.hs[-2],self.gRecurrent)

			bp(self.delta,self.w1,self.gw1,self.gb1,self.inputs[-1],self.gInput)
			#print(np.argmax(asarray(self.inputs[-1])))

			self.hs.pop()
			self.inputs.pop()
			self.outputs.pop()
			t.pop()

	def updateWeights(self):
		mclip(self.gwr)
		mclip(self.gw2)
		mclip(self.gw1)
		#print('gw2',asarray(self.gw2))
		#print('gwr',asarray(self.gwr))
		#print('gw1',asarray(self.gw1))
		update_weights(self.w2,self.gw2,self.lr)
		update_weights(self.b2,self.gb2,self.lr)
		update_weights(self.w1,self.gw1,self.lr)
		update_weights(self.b1,self.gb1,self.lr)
		update_weights(self.wr,self.gwr,self.lr)
		update_weights(self.br,self.gbr,self.lr)
		
		#print(asarray(self.wr))
		self.forget()

	def train(self,ds,epochs,lr=0.001,decay=0.99):
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		self.lr = lr
		acc = 0
		w = 0
		time = 0.
		wps = 0.
		for epoch in range(epochs):
			start = timer()
			correct = 0
			count = 0
			for seq in range(len(ds)-1):
				x = ds[seq]
				targets = []
				st = timer()
				for t in range(len(x)-1):
					count += 1
					w += 1
					inval = encode(x[t])
					#print('inval',asarray(inval))
					tarval = encode(x[t+1])
					#print('tarval',asarray(tarval))
					self.forward(inval)
					#print('output',asarray(self.output))
					targets.append(mcopy(tarval))
					if most_similar(self.outputs[-1]) == x[t+1]:
						#print('correct')
						correct += 1
				#print(targets)
				acc = float(correct)/float(count)
				self.bptt(targets)
				#print('output',asarray(self.outputs[-1]))
				#print(asarray(self.outputs[-1]))
				#print('Outputs:',decode(self.outputs[-2]),decode(self.outputs[-1]),'Input',x[-2],'Target',decode(x[-1]))
				#print('gw2',self.gw2.asarray(),'gb2',self.gb2.asarray(),'iifog',cm.sum(self.hidden_layer.gi_IFOG,axis=1).sum(axis=0).asarray(),'hifog',self.hidden_layer.hm1_IFOG.asarray())
				self.updateWeights()
				time += timer()-st
				wps = float(w)/time
				#print('wps:',wps,"eta:",(float(utils.total)/wps)/60,'min')
				#if (seq % 100 == 0) and (self.lr > 0.005):
					#self.lr = self.lr * decay
			time = timer() - start
			sent = [ds[10][0]]
			for i in range(15):
				x = encode(sent[-1])
				y = self.forward(x)
				sent.append(most_similar(y))
			self.forget()
			print('Trained Epoch:',epoch+1,"With Accuracy:",acc, 'in', time, 'seconds', 'Learning Rate:',self.lr, 'wps',wps)
			print('Generated Sentence:',sent)

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
		mzero(self.delta)
		mzero(self.gRecurrent)
		mzero(self.gInput)
		self.hs = [zeros([1,self.layers[1]])]
		self.inputs=[]

	def forget(self):
		self.reset_grads()
		self.reset_activations()

ds = load_sentences_data('../data/ptb.train.short.txt',use_embeddings=True)

net = rnn([500,1000,500])

net.train(ds,65)
