import numpy as np 
from timeit import default_timer as timer
import utils
from utils import *

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
		fp(x,self.w1,self.b1,self.h)
		fp(self.hs[-1],self.wr,self.br,self.r)
		mmadd(self.h,self.r,self.h)
		mtanh(self.h,self.h)
		fp(self.h,self.w2,self.b2,self.output)
		msoftmax(self.output,self.output)
		self.inputs.append(mcopy(x))
		self.hs.append(mcopy(self.h))
		self.outputs.append(mcopy(self.output))
		return self.output

	def bptt(self,t):
		for q in range(len(t)-2):
			mmsubtract(self.outputs[-1],t[-1],self.gOutput)
			bp(self.gOutput,self.w2,self.gw2,self.gb2,self.hs[-1],self.delta)
			
			mclip(self.delta)
			mtanh_deriv(self.delta,self.hs[-1],self.delta)
			mclip(self.delta)

			bp(self.delta,self.w1,self.gw1,self.gb1,self.inputs[-1],self.gInput)
			#print(np.argmax(asarray(self.inputs[-1])))
			bp(self.delta,self.wr,self.gwr,self.gbr,self.hs[-2],self.gRecurrent)
			mclip(self.gbr)
			mclip(self.gwr)
			mclip(self.gw1)
			mclip(self.gb1)
			mclip(self.gw2)
			mclip(self.gb2)

			self.hs.pop()
			self.inputs.pop()
			self.outputs.pop()
			t.pop()

		#print(self.gwr.asarray())

	def updateWeights(self):
		#print(asarray(self.gwr))
		update_weights(self.w2,self.gw2,self.lr)
		update_weights(self.b2,self.gb2,self.lr)
		update_weights(self.w1,self.gw1,self.lr)
		update_weights(self.b1,self.gb1,self.lr)
		update_weights(self.wr,self.gwr,self.lr)
		update_weights(self.br,self.gbr,self.lr)
		#print(asarray(self.w1))
		self.forget()

	def train(self,ds,epochs,lr=0.01,decay=0.99):
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
					tarval = encode(x[t+1])
					self.forward(inval)
					targets.append(tarval)
					if utils.vocab[decode(x[t+1])] == asarray(self.outputs[-1]).argmax(axis=1):
						correct += 1
				#print(targets)
				acc = float(correct)/float(count)
				self.bptt(targets)
				
				#print('Outputs:',utils.decode(self.outputs[-2]),utils.decode(self.outputs[-1]),'Input',x[-2],'Target',utils.decode(targets[-1]))
				#print('gw2',self.gw2.asarray(),'gb2',self.gb2.asarray(),'iifog',cm.sum(self.hidden_layer.gi_IFOG,axis=1).sum(axis=0).asarray(),'hifog',self.hidden_layer.hm1_IFOG.asarray())
				self.updateWeights()
				time += timer()-st
				wps = float(w)/time
				#print('wps:',wps,"eta:",(float(utils.total)/wps)/60,'min')
				#if (seq % 100 == 0) and (self.lr > 0.005):
					#self.lr = self.lr * decay
				self.reset_activations()
			time = timer() - start
			sent = [ds[10][0]]
			for i in range(15):
				x = encode(sent[-1])
				y = self.forward(x)
				sent.append(decode(y))
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
		self.hs = [zeros([1,self.layers[1]])]
		self.inputs=[]

	def forget(self):
		self.reset_grads()
		self.reset_activations()

ds = load_sentences_data('../data/ptb.train.short.txt')

net = rnn([utils.word_idx,1000,utils.word_idx])

net.train(ds,65)
