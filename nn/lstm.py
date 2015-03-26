import cudamat as cm
from cudamat import learn as cl
import numpy as np
import copy
from timeit import default_timer as timer
import math
import pickle
import utils
from utils import *
import sys

cm.shutdown()
cm.init()

class lstm(object):
	def __init__(self, layers,uplim=15,lowlim=-15):
		super(lstm, self).__init__()
		
		self.layers = layers
		self.outputs = []
		self.uplim = uplim
		self.lowlim = lowlim

		# Build Netowrk

		#LSTM Layer

		self.hidden_layer = lstm_layer(layers,uplim=self.uplim,lowlim=self.lowlim)

		#Hidden to Output Weights

		self.w2 = init_weights([layers[-2],layers[-1]])

		self.b2 = init_weights([1,layers[-1]])

		self.gw2 = zeros(self.w2.shape)
		self.gb2 = zeros(self.b2.shape)
		self.gOutput = zeros([1,self.layers[2]])

		self.forget()
		self.updates_tm1 = [self.gw2,self.gb2]

	def forward(self,x):
		h = self.hidden_layer.forward(x)
		output = zeros(h.shape)
		msoftmax(h,output)
		self.inputs.append(mcopy(x))
		self.outputs.append(mcopy(output))
		return self.output


	def bptt(self,t):
		#Set T+1 activations to 0
		self.hidden_layer.prev_outputs.append(zeros([1,self.layers[1]]))
		self.hidden_layer.prev_states.append(zeros([1,self.layers[1]]))
		self.hidden_layer.prev_gates.append(zeros([1,self.layers[1]*4]))
		self.hidden_layer.prev_fgates.append(zeros([1,self.layers[1]*4]))

		for _ in range(len(t)-1,-1,-1):	
			#print('Delta',self.delta.asarray())
			mzero(self.gOutput)
			mmsubtract(self.outputs[_+1],t[_],self.gOutput)
			delta = bp(self.gOutput,self.w2,self.gw2,self.gb2,self.hidden_layer.prev_outputs[_+1])
			self.hidden_layer.backward(delta,_+1)

	def updateWeights(self):
		msmult(self.gw2,self.lr,self.gw2)
		msmult(self.updates_tm1[0],0.9,self.updates_tm1[0])
		mmadd(self.gw2,self.updates_tm1[0],self.gw2)
		mmsubtract(self.w2,self.gw2,self.w2)

		msmult(self.gb2,self.lr,self.gb2)
		msmult(self.updates_tm1[1],0.9,self.updates_tm1[1])
		mmadd(self.gb2,self.updates_tm1[1],self.gb2)
		mmsubtract(self.b2,self.gb2,self.b2)
		
		self.hidden_layer.updateWeights(self.lr)
		self.updates_tm1 = [mcopy(self.gw2),mcopy(self.gb2)]
		self.forget()

	def train(self,ds,epochs,batch_size=1,lr=0.05,decay=0.99):
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		self.lr = lr
		self.last_best_acc = 0
		acc = 0
		self.last_best_model = []
		for epoch in range(epochs):
			start = timer()
			correct = 0
			count = 0
			for seq in range(len(ds)-1):
				x = ds[seq]
				targets = []
				for t in range(len(x)-1):
					count += 1
					inval = encode(x[t])
					tarval = encode(x[t+1])
					self.forward(inval)
					targets.append(tarval)
					if vocab[x[t+1]] == asarray(self.outputs[-1]).argmax(axis=1):
						correct += 1
				#print(targets)
				acc = float(correct)/float(count)
				if acc > self.last_best_acc:
					self.last_best_acc = acc
					self.last_best_model = [asarray(self.w2),asarray(self.b2)]
					self.last_best_model.append(asarray(self.hidden_layer.i_IFOG))
					self.last_best_model.append(asarray(self.hidden_layer.hm1_IFOG))
					self.lr = self.lr*decay
				self.bptt(targets)
				if seq % batch_size == 0:
					#rint('Outputs:',utils.decode(self.outputs[-2].asarray()),utils.decode(self.outputs[-1].asarray()),'Input',x[-2],'Target',utils.decode(targets[-1].asarray()))
					#print('gw2',self.gw2.asarray(),'gb2',self.gb2.asarray(),'iifog',cm.sum(self.hidden_layer.gi_IFOG,axis=1).sum(axis=0).asarray(),'hifog',self.hidden_layer.hm1_IFOG.asarray())
					self.updateWeights()
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
			print('Trained Epoch:',epoch+1,"With Accuracy:",acc, 'in', time, 'seconds', 'Learning Rate:',self.lr)
			print('Generated Sentence:',sent)


	def reset_grads(self):
		mzero(self.gw2)
		mzero(self.gb2)
		mzero(self.gOutput)
		mzero(self.delta)

	def reset_activations(self):
		#print("MAIN RESET")
		#self.dmasks = [cm.CUDAMatrix(np.random.binomial(n=1,p=0.8,size=self.w2.shape)),cm.CUDAMatrix(np.random.binomial(n=1,p=0.8,size=self.b2.shape))]
		mzero(self.output)
		self.outputs = [zeros(1,self.layers[-1])]
		mzero(self.h)
		self.hs = [zeros(1,self.layers[-1])]
		self.inputs=[zeros(1,self.layers[0])]
		self.hidden_layer.reset_activations()

	def forget(self):
		self.reset_grads()
		self.reset_activations()
		self.hidden_layer.forget()

	def last_best(self):
		self.w2 = cuda.to_device(self.last_best_model[0])
		self.b2 = cuda.to_device(self.last_best_model[1])
		self.hidden_layer.i_IFOG = cuda.to_device(self.last_best_model[2])
		self.hidden_layer.hm1_IFOG = cuda.to_device(self.last_best_model[3])
		
class lstm_layer(object):
	def __init__(self, layers,uplim=1,lowlim=-1):
		super(lstm_layer, self).__init__()

		self.layers = layers

		#Input Gate Weights

		self.i_IFOG = init_weights([self.layers[0],self.layers[1]*4])

		self.hm1_IFOG = init_weights([self.layers[1],self.layers[1]*4])

		self.uplim = uplim
		self.lowlim = lowlim
		self.updates_tm1 = [zeros([self.layers[0],self.layers[1]*4]),zeros([self.layers[1],self.layers[1]*4])]
		temp = zeros([1,self.layers[1]])
		i = zeros([self.layers[0],self.layers[1]])
		f = zeros([self.layers[0],self.layers[1]])
		o = zeros([self.layers[0],self.layers[1]])
		g = zeros([self.layers[0],self.layers[1]])
		self.gates=[i,f,o,g]
		self.states = zeros([1,self.layers[1]])
		self.output = zeros([1,self.layers[1]])
		self.recurrentGrad = zeros([1,self.layers[1]*4])
		self.ec = zeros([1,self.layers[1]])
		self.es = zeros([1,self.layers[1]])
		self.go = zeros([1,self.layers[1]])
		self.gg = zeros([1,self.layers[1]])
		self.gf = zeros([1,self.layers[1]])
		self.gi = zeros([1,self.layers[1]])
		self.ggates = zeros([1,self.layers[1]*4])
		self.gi_IFOG = zeros([self.layers[0],self.layers[1]*4])
		self.ghm1_IFOG = zeros([self.layers[1]*4,self.layers[1]*4])

		self.forget()

	def forward(self,x):
		mzero(temp)
		i = mzero(self.gates[0])
		f = mzero(self.gates[1])
		o = mzero(self.gates[2])
		g = mzero(self.gates[3])

		mmprod(x,self.i_IFOG,temp)
		mmprod(self.prev_outputs[-1],self.hm1_IFOG,temp)
		ifog_split(temp,self.gates)
		self.prev_gates.append([mcopy(i),mcopy(f),mcopy(o),mcopy(g)])

		i = mzero(self.gates[0])
		f = mzero(self.gates[1])
		o = mzero(self.gates[2])
		g = mzero(self.gates[3])

		ifog_activate(temp,self.gates)
		
		mzero(self.states)
		mmult(i,g,states)
		mzero(temp)
		mmult(f,self.prev_states[-1],temp)
		mmadd(states,temp,states)

		mzero(self.output)
		mtanh(states,self.output)
		mmult(self.output,o,self.output)

		self.prev_outputs.append(self.output)
		self.prev_states.append(states)
		self.prev_fgates.append([mcopy(i),mcopy(f),mcopy(o),mcopy(g)])
		self.inputs.append(x)
		#print(self.output.asarray())
		return self.output

	def backward(self,grad,t):
		#print('prev_gc',self.prev_gc[-1].asarray())

		mzero(temp)
		mzero(recurrentGrad)
		#print(temp.asarray())

		mmprod(self.prev_ggates[-1],mtranspose(self.hm1_IFOG),self.recurrentGrad)

		i = self.prev_gates[t][0]
		f = self.prev_gates[t][1]
		o = self.prev_gates[t][2]
		g = self.prev_gates[t][3]

		fi = self.prev_fgates[t][0]
		ff = self.prev_fgates[t][1]
		fo = self.prev_fgates[t][2]
		fg = self.prev_fgates[t][3]

		s = self.prev_states[t]
		s_tm1 = self.prev_states[t-1]

		es_tp1 = self.prev_es[-1]

		ff_tp1 = self.prev_fgates[t+1][1]

		mzero(self.ec)
		mmadd(self.ec,grad,self.ec)
		mmadd(self.ec,self.recurrentGrad,self.ec)

		#Loss wrt Cell State
		mtanh_deriv(s,self.es)
		
		mmult(self.es,fo,self.es)
		mmult(self.es,ec,self.es)
		
		mzero(self.temp)

		mmult(ff_tp1,es_tp1,self.temp)
		mmadd(self.es,self.temp,self.es)

		#Gradient at Output Gates
		mtanh(s,self.go)
		mmult(self.go,self.ec,self.go)
		mzero(self.temp)
		msigmoid_deriv(o,self.temp)
		mmult(self.go,self.temp,self.go)

		#Gradient at Cell Input
		mtanh_deriv(g,self.gg)
		mzero(self.temp)
		mmult(fi,es,self.temp)
		mmult(self.gg,self.temp,self.gg)

		#Gradient at Forget Gate
		mmult(s_tm1,self.es,self.gf)
		mzero(self.temp)
		msigmoid_deriv(f,self.temp)
		mmult(self.gf,self.temp,self.gf)

		#Gradient at Input Gate
		mtanh(g,self.gi)
		mmult(self.gi,es,self.gi)
		mzero(self.temp)
		msigmoid_deriv(i,self.temp)
		mmult(self.gi,self.temp,self.gi)

		#self.clip(es)
		self.prev_es.append(mcopy(self.es))

		ifog_build(self.ggates,[gi,gf,go,gg])

		#self.clip(ggates)
		self.prev_ggates.append(mcopy(self.ggates))

		#self.gradInput = cm.dot(ggates,self.i_IFOG.T)
		#self.clip(self.gradInput)

		#Accumulate Gradients

		mmprod(mtranspose(self.inputs[t]),self.ggates,self.gi_IFOG)
		mmprod(mtranspose(self.prev_outputs[t-1]),self.ggates,ghm1_IFOG)

		mclip(self.gi_IFOG)
		mclip(self.ghm1_IFOG)

	def updateWeights(self,lr):
		#self.clip(self.ghm1_IFOG)
		msmult(self.gi_IFOG,lr,self.gi_IFOG)
		msmult(self.updates_tm1[0],0.9,self.updates_tm1[0])
		mmadd(self.gi_IFOG,self.updates_tm1[0],self.gi_IFOG)
		mmsubtract(self.i_IFOG,self.gi_IFOG,self.i_IFOG)

		msmult(self.ghm1_IFOG,lr,self.ghm1_IFOG)
		msmult(self.updates_tm1[1],0.9,self.updates_tm1[1])
		mmadd(self.ghm1_IFOG,self.updates_tm1[0],self.ghm1_IFOG)
		mmsubtract(self.hm1_IFOG,self.ghm1_IFOG,self.hm1_IFOG)

		self.updates_tm1 = [self.gi_IFOG,self.ghm1_IFOG]
		#print(self.i_IFOG.asarray())

	def forget(self):
		self.reset_activations()
		self.reset_grads()

	def reset_activations(self):
		#print('RESETTING')
		self.prev_states = [zeros([1,self.layers[1]])]
		self.prev_outputs =[zeros([1,self.layers[1]])]
		self.prev_gates =[zeros([1,self.layers[1]*4])]
		self.prev_fgates =[zeros([1,self.layers[1]*4])]
		self.prev_ggates =[zeros([1,self.layers[1]*4])]
		self.inputs = [zeros([1,self.layers[0]])]

		self.prev_es = [zeros([1,self.layers[1]])]

	def reset_grads(self):
		#print('Resetting Grads')
		mzero(self.gi_IFOG)
		mzero(self.ghm1_IFOG)

ds = load_data('../data/ptb.train.txt')

n_tokens = utils.word_idx
net = lstm([n_tokens,400,n_tokens])

start = timer()
print('Starting Training')
net.train(ds,200)
time  = timer() - start
print('Training Time:',time)

net.last_best()

net.forget()

"""
sent = [enc.inverse_transform(ds[0][12][0].asarray())]
for i in range(30):
	x = cm.CUDAMatrix(enc.transform([sent[-1]]))
	y = net.forward(x)
	sent.append(enc.inverse_transform(y.asarray())[0])

print(sent)
"""
