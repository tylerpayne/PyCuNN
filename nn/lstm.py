import cudamat as cm
from cudamat import learn as cl
import numpy as np
import copy
from sklearn import preprocessing as prepro
from timeit import default_timer as timer
import math
import pickle
import utils
#from nltk.corpus import brown
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

		self.w2 = utils.init_weights(layers[-2],layers[-1])

		self.b2= utils.init_weights(1,layers[-1])

		self.forget()
		self.updates_tm1 = [self.gw2,self.gb2]

	def forward(self,x):
		self.h = self.hidden_layer.forward(x)
		logits = cm.exp(cm.dot(self.h,self.w2)).add(self.b2)
		self.output = logits.mult_by_col(cm.pow(cm.sum(logits,axis=1),-1))
		self.inputs.append(x)
		self.outputs.append(self.output)
		return self.output

	def bptt(self,t):
		#Set T+1 activations to 0
		self.hidden_layer.prev_outputs.append(utils.zeros(1,self.layers[1]))
		self.hidden_layer.prev_states.append(utils.zeros(1,self.layers[1]))
		self.hidden_layer.prev_gates.append(utils.zeros(1,self.layers[1]*4))
		self.hidden_layer.prev_fgates.append(utils.zeros(1,self.layers[1]*4))

		for _ in range(len(t)-1,-1,-1):	
			#print('Delta',self.delta.asarray())
			self.outputs[_+1].subtract(t[_],target=self.gOutput)
			self.gw2.add_dot(self.hidden_layer.prev_outputs[_+1].T,self.gOutput)
			self.gb2.add_sums(self.gOutput,axis=0)
			self.clip(self.gw2)
			self.clip(self.gb2)

			self.delta = cm.dot(self.gOutput,self.w2.T)
			self.clip(self.delta)
			self.hidden_layer.backward(self.delta,_+1)

	def clip(self,param):
		norm = param.euclid_norm()
		if norm > self.uplim:
			param.mult(float(self.uplim) / norm)

	def updateWeights(self):
		self.w2.subtract(self.gw2.mult(self.lr).add(self.updates_tm1[0].mult(0.9)))
		self.b2.subtract(self.gb2.mult(self.lr).add(self.updates_tm1[1].mult(0.9)))
		self.hidden_layer.updateWeights(self.lr)
		self.updates_tm1 = [self.gw2,self.gb2]
		self.forget()

	def train(self,ds,epochs,enc,batch_size=1,lr=0.1,decay=0.99):
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
				for t in range(len(x)):
					count += 1
					self.forward(cm.CUDAMatrix(enc.transform([x[t][0]])))
					targets.append(cm.CUDAMatrix(enc.transform([x[t][1]])))
					if x[t][1].argmax(axis=1) == self.outputs[-1].argmax(axis=1).asarray()[0][0]:
						correct += 1
				#print(targets)
				acc = float(correct)/float(count)
				if acc > self.last_best_acc:
					self.last_best_acc = acc
					self.last_best_model = [self.w2.asarray(),self.b2.asarray()]
					self.last_best_model.append(self.hidden_layer.i_IFOG.asarray())
					self.last_best_model.append(self.hidden_layer.hm1_IFOG.asarray())
					self.lr = self.lr*decay
				self.bptt(targets)
				if seq % batch_size == 0:
					#print('Outputs:',enc.inverse_transform(self.outputs[-2].asarray()),enc.inverse_transform(self.outputs[-1].asarray()),'Input',enc.inverse_transform(x[-1][0].asarray()),'Target',enc.inverse_transform(targets[-1].asarray()))
					#print('gw2',self.gw2.asarray(),'gb2',self.gb2.asarray(),'iifog',cm.sum(self.hidden_layer.gi_IFOG,axis=1).sum(axis=0).asarray(),'hifog',self.hidden_layer.hm1_IFOG.asarray())
					self.updateWeights()
				#if (seq % 100 == 0) and (self.lr > 0.005):
					#self.lr = self.lr * decay
				self.reset_activations()
			time = timer() - start
			sent = [enc.inverse_transform(ds[10][0][0].asarray())]
			for i in range(15):
				x = cm.CUDAMatrix(enc.transform([sent[-1]]))
				y = self.forward(x)
				sent.append(enc.inverse_transform(y.asarray())[0])
			self.forget()
			print('Trained Epoch:',epoch+1,"With Accuracy:",acc, 'in', time, 'seconds', 'Learning Rate:',self.lr)
			print('Generated Sentence:',sent)


	def reset_grads(self):
		self.gw2 = utils.zeros(self.layers[-2],self.layers[-1])
		self.gb2 = utils.zeros(1,self.layers[-1])
		self.gOutput = utils.zeros(1,self.layers[-1])
		self.delta = utils.zeros(1,self.layers[-2])

	def reset_activations(self):
		#print("MAIN RESET")
		#self.dmasks = [cm.CUDAMatrix(np.random.binomial(n=1,p=0.8,size=self.w2.shape)),cm.CUDAMatrix(np.random.binomial(n=1,p=0.8,size=self.b2.shape))]
		self.output = utils.zeros(1,self.layers[-1])
		self.outputs = [utils.zeros(1,self.layers[-1])]
		self.h = utils.zeros(1,self.layers[-2])
		self.hs = [utils.zeros(1,self.layers[-1])]
		self.inputs=[utils.zeros(1,self.layers[0])]
		self.hidden_layer.reset_activations()

	def forget(self):
		self.reset_grads()
		self.reset_activations()
		self.hidden_layer.forget()

	def last_best(self):
		self.w2 = cm.CUDAMatrix(self.last_best_model[0])
		self.b2 = cm.CUDAMatrix(self.last_best_model[1])
		self.hidden_layer.i_IFOG = cm.CUDAMatrix(self.last_best_model[2])
		self.hidden_layer.hm1_IFOG = cm.CUDAMatrix(self.last_best_model[3])
		
class lstm_layer(object):
	def __init__(self, layers,uplim=1,lowlim=-1):
		super(lstm_layer, self).__init__()

		self.layers = layers

		#Input Gate Weights

		self.i_IFOG = utils.init_weights(self.layers[0],self.layers[1]*4)

		self.hm1_IFOG = utils.init_weights(self.layers[1],self.layers[1]*4)

		self.uplim = uplim
		self.lowlim = lowlim
		self.updates_tm1 = [utils.zeros(self.layers[0],self.layers[1]*4),utils.zeros(self.layers[1],self.layers[1]*4)]
		self.forget()

	def forward(self,x):
		temp = utils.zeros(1,self.layers[1])

		gates = cm.dot(x,self.i_IFOG).add_dot(self.prev_outputs[-1],self.hm1_IFOG)
		self.prev_gates.append(gates)
		ifo = gates.get_col_slice(0,self.layers[1]*3)
		g = gates.get_col_slice(self.layers[1]*3,self.layers[1]*4)

		cm.sigmoid(ifo)
		cm.tanh(g)

		i = gates.get_col_slice(0,self.layers[1])
		f = gates.get_col_slice(self.layers[1],self.layers[1]*2)
		o = gates.get_col_slice(self.layers[1]*2,self.layers[1]*3)
		
		states = utils.zeros(1,self.layers[1])
		i.mult(g, target = states)
		f.mult(self.prev_states[-1], target = temp)
		states.add(temp)

		self.output = utils.zeros(1,self.layers[1])
		cm.tanh(states, target = self.output)
		self.output.mult(o)

		self.prev_outputs.append(self.output)
		self.prev_states.append(states)
		self.prev_fgates.append(gates)
		self.inputs.append(x)
		#print(self.output.asarray())
		return self.output

	def backward(self,grad,t):
		#print('prev_gc',self.prev_gc[-1].asarray())

		temp = utils.zeros(1,self.layers[1])
		#print(temp.asarray())

		recurrentGrad = cm.dot(self.prev_ggates[-1],self.hm1_IFOG.T)

		i = self.prev_gates[t].get_col_slice(0,self.layers[1])
		f = self.prev_gates[t].get_col_slice(self.layers[1],self.layers[1]*2)
		o = self.prev_gates[t].get_col_slice(self.layers[1]*2,self.layers[1]*3)
		g = self.prev_gates[t].get_col_slice(self.layers[1]*3,self.layers[1]*4)

		fi = self.prev_fgates[t].get_col_slice(0,self.layers[1])
		ff = self.prev_fgates[t].get_col_slice(self.layers[1],self.layers[1]*2)
		fo = self.prev_fgates[t].get_col_slice(self.layers[1]*2,self.layers[1]*3)
		fg = self.prev_fgates[t].get_col_slice(self.layers[1]*3,self.layers[1]*4)

		s = self.prev_states[t]
		s_tm1 = self.prev_states[t-1]

		es_tp1 = self.prev_es[-1]

		ff_tp1 = self.prev_fgates[t+1].get_col_slice(self.layers[1],self.layers[1]*2)

		ec = utils.zeros(1,self.layers[1])
		ec.add(grad).add(recurrentGrad)

		#Loss wrt Cell State
		temp = cm.CUDAMatrix(np.ones(s.shape))
		es = utils.zeros(1,self.layers[1])
		temp.subtract(cm.pow(cm.tanh(s),2),target=es)
		#print(es.asarray())
		es.mult(fo).mult(ec)
		ff_tp1.mult(es_tp1,target=temp)
		es.add(temp)

		#Gradient at Output Gates
		go = utils.zeros(1,self.layers[1])
		cm.tanh(s,target = go)
		cl.mult_by_sigmoid_deriv(go,o)
		go.mult(ec)

		#Gradient at Cell Input
		temp = cm.CUDAMatrix(np.ones(s.shape))
		gg = temp.subtract(cm.pow(cm.tanh(g),2))
		fi.mult(es,target=gg)
		cl.mult_by_sigmoid_deriv(gg,g)

		#Gradient at Forget Gate
		gf = utils.zeros(1,self.layers[1])
		s_tm1.mult(es,target=gf)
		cl.mult_by_sigmoid_deriv(gf,f)

		#Gradient at Input Gate
		gi = utils.zeros(1,self.layers[1])
		cm.tanh(g,target=gi)
		gi.mult(es)
		cl.mult_by_sigmoid_deriv(gi,i)

		#self.clip(es)
		self.prev_es.append(es)

		ggates = utils.zeros(1,self.layers[1]*4)
		ggates.set_col_slice(0,self.layers[1],gi)
		ggates.set_col_slice(self.layers[1],self.layers[1]*2,gf)
		ggates.set_col_slice(self.layers[1]*2,self.layers[1]*3,go)
		ggates.set_col_slice(self.layers[1]*3,self.layers[1]*4,gg)

		#self.clip(ggates)
		self.prev_ggates.append(ggates)

		#self.gradInput = cm.dot(ggates,self.i_IFOG.T)
		#self.clip(self.gradInput)

		#Accumulate Gradients

		di = self.gi_IFOG.get_col_slice(0,self.layers[1])
		df = self.gi_IFOG.get_col_slice(self.layers[1],self.layers[1]*2)
		do = self.gi_IFOG.get_col_slice(self.layers[1]*2,self.layers[1]*3)
		dg = self.gi_IFOG.get_col_slice(self.layers[1]*3,self.layers[1]*4)

		di.add_dot(self.inputs[t].T,gi)
		df.add_dot(self.inputs[t].T,gf)
		do.add_dot(self.inputs[t].T,go)
		dg.add_dot(self.inputs[t].T,gg)

		dhi = self.ghm1_IFOG.get_col_slice(0,self.layers[1])
		dhf = self.ghm1_IFOG.get_col_slice(self.layers[1],self.layers[1]*2)
		dho = self.ghm1_IFOG.get_col_slice(self.layers[1]*2,self.layers[1]*3)
		dhg = self.ghm1_IFOG.get_col_slice(self.layers[1]*3,self.layers[1]*4)

		dhi.add_dot(self.prev_outputs[t-1].T,gi)
		dhf.add_dot(self.prev_outputs[t-1].T,gf)
		dho.add_dot(self.prev_outputs[t-1].T,go)
		dhg.add_dot(self.prev_outputs[t-1].T,gg)

		self.clip(self.gi_IFOG)
		self.clip(self.ghm1_IFOG)

	def clip(self,param):
		norm = param.euclid_norm()
		if norm > self.uplim:
			param.mult(float(self.uplim) / norm)

	def updateWeights(self,lr):
		#self.clip(self.ghm1_IFOG)
		self.i_IFOG.subtract(self.gi_IFOG.mult(lr).add(self.updates_tm1[0].mult(0.9)))
		self.hm1_IFOG.subtract(self.ghm1_IFOG.mult(lr).add(self.updates_tm1[1].mult(0.9)))
		self.updates_tm1 = [self.gi_IFOG,self.ghm1_IFOG]
		#print(self.i_IFOG.asarray())

	def forget(self):
		self.reset_activations()
		self.reset_grads()

	def reset_activations(self):
		#print('RESETTING')
		self.prev_states = [utils.zeros(1,self.layers[1])]
		self.prev_outputs =[utils.zeros(1,self.layers[1])]
		self.prev_gates =[utils.zeros(1,self.layers[1]*4)]
		self.prev_fgates =[utils.zeros(1,self.layers[1]*4)]
		self.prev_ggates =[utils.zeros(1,self.layers[1]*4)]
		self.inputs = [utils.zeros(1,self.layers[0])]
		self.gradInput = utils.zeros(1,self.layers[0])

		self.prev_es = [utils.zeros(1,self.layers[1])]

	def reset_grads(self):
		#print('Resetting Grads')
		self.gi_IFOG = utils.zeros(self.layers[0],self.layers[1]*4)
		self.ghm1_IFOG = utils.zeros(self.layers[1],self.layers[1]*4)

start = timer()
print('Loading Text')
'''words = list(brown.words())
sentences = list(brown.sents())
total = len(sentences)
'''

with open('../data/ptb.train.txt','r+') as doc:
	f = doc.read()
	sentences = f.split('\n')
	words = f.split(' ')
enc = prepro.LabelBinarizer()
enc.fit(words)
#ds= utils.encode_sentences(enc,,sentences)
ds = []
del(sentences[-1])
for z in sentences:
	s = z.split(' ')
	del s[0]
	del s[-1]
	sent = []
	for w in range(len(s)-2):
		sent.append([s[w],s[w+1]])
	ds.append(sent)
print('Built Dataset in ', timer() - start, "s")


#ds = np.array(ds.tolist())
#print(ds.shape)
#print(ds[0][0][1])


n_tokens = enc.classes_.shape[0]
net = lstm([n_tokens,1000,n_tokens])

start = timer()
print('Starting Training')
net.train(ds,200,enc)
time  = timer() - start
print('Training Time:',time)

net.last_best()

net.forget()
sent = [enc.inverse_transform(ds[0][12][0].asarray())]
for i in range(30):
	x = cm.CUDAMatrix(enc.transform([sent[-1]]))
	y = net.forward(x)
	sent.append(enc.inverse_transform(y.asarray())[0])

print(sent)

