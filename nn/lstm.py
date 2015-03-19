import cudamat as cm
from cudamat import learn as cl
import numpy as np
import copy
from sklearn import preprocessing as prepro
import timeit
import math

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

		self.w2 = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[-2] + self.layers[-1])),
			high=np.sqrt(6. / (self.layers[-2] + self.layers[-1])),
			size = (self.layers[-2],self.layers[-1])
		))

		self.b2= cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[-2] + self.layers[-1])),
			high=np.sqrt(6. / (self.layers[-2] + self.layers[-1])),
			size = (1,self.layers[-1])
		))

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
		#print('bptt')
		#Set T+1 activations to 0
		self.hidden_layer.prev_outputs.append(cm.CUDAMatrix(np.zeros(self.hidden_layer.prev_outputs[-1].shape)))
		self.hidden_layer.prev_states.append(cm.CUDAMatrix(np.zeros(self.hidden_layer.prev_states[-1].shape)))
		self.hidden_layer.prev_gates.append(cm.CUDAMatrix(np.zeros(self.hidden_layer.prev_gates[-1].shape)))
		self.hidden_layer.prev_fgates.append(cm.CUDAMatrix(np.zeros(self.hidden_layer.prev_fgates[-1].shape)))

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


	def train(self,ds,epochs,enc,batch_size=10,lr=0.01,decay=0.99):
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		self.lr = lr/batch_size
		self.last_best_acc = 0
		acc = 0
		self.last_best_model = []
		for epoch in range(epochs):
			correct = 0
			count = 0
			#print(seq_len)
			for seq in range(len(ds)-1):
				#print('seq',seq)
				x = ds[seq]
				targets = []
				for t in range(len(x)):
					count += 1
					self.forward(x[t][0])
					targets.append(x[t][1])
					if x[t][1].argmax(axis=1).asarray()[0][0] == self.outputs[-1].argmax(axis=0).asarray()[0][0]:
						correct += 1
				#print(targets)
				acc = float(correct)/float(count)
				if acc > self.last_best_acc:
					self.last_best_acc = acc
					self.last_best_model = [self.w2.asarray(),self.b2.asarray()]
					self.last_best_model.append(self.hidden_layer.i_IFOG.asarray())
					self.last_best_model.append(self.hidden_layer.hm1_IFOG.asarray())
				self.bptt(targets)
				if seq % batch_size == 0:
					print('Outputs:',enc.inverse_transform(self.outputs[-2].asarray()),enc.inverse_transform(self.outputs[-1].asarray()),'Input',enc.inverse_transform(x[-1][0].asarray()),'Target',enc.inverse_transform(targets[-1].asarray()))
					#print('gw2',self.gw2.asarray(),'gb2',self.gb2.asarray(),'iifog',cm.sum(self.hidden_layer.gi_IFOG,axis=1).sum(axis=0).asarray(),'hifog',self.hidden_layer.hm1_IFOG.asarray())
					self.updateWeights()
					#self.lr = self.lr * decay
				self.reset_activations()
			
			print('Trained Epoch:',epoch+1,"With Accuracy:",acc)


	def reset_grads(self):
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[-2],self.layers[-1]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[-1]]))
		self.gOutput = cm.CUDAMatrix(np.zeros([1,self.layers[-1]]))
		self.delta = cm.CUDAMatrix(np.zeros([1,self.layers[-2]]))

	def reset_activations(self):
		#print("MAIN RESET")
		#self.dmasks = [cm.CUDAMatrix(np.random.binomial(n=1,p=0.8,size=self.w2.shape)),cm.CUDAMatrix(np.random.binomial(n=1,p=0.8,size=self.b2.shape))]
		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[-1]]))
		self.outputs = [cm.CUDAMatrix(np.zeros([1,self.layers[-1]]))]
		self.h = cm.CUDAMatrix(np.zeros([1,self.layers[-2]]))
		self.hs = [cm.CUDAMatrix(np.zeros([1,self.layers[-1]]))]
		self.inputs=[cm.CUDAMatrix(np.zeros([1,self.layers[0]]))]
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

	def gradient_checking(self,ds,epochs,enc,batch_size=1,lr=0.05,decay=0.99):
		self.lr = lr/batch_size
		self.last_best_acc = 0
		acc = 0
		e= 0.001
		for epoch in range(epochs):
			correct = 0
			count = 0
			#print(seq_len)
			for seq in range(1000):
				#print('seq',seq)
				x = ds[seq][0][0]
				targets = [ds[seq][0][1]]
				count += 1
				self.forward(x)
				#if x.argmax(axis=1).asarray()[0][0] == self.outputs[-1].argmax(axis=1).asarray()[0][0]:
					#correct += 1
				#print(targets)
				#acc = float(correct)/float(count)
				self.bptt(targets)
				gvals = [self.gw2.asarray(),self.gb2.asarray(),self.hidden_layer.gi_IFOG.asarray(),self.hidden_layer.ghm1_IFOG.asarray()]
				self.forget()

				#self.w2.add(e)
				#self.hidden_layer.i_IFOG.add(e)
				self.hidden_layer.hm1_IFOG.add(e)

				self.forward(x)

				plus_loss = cm.CUDAMatrix(np.zeros(self.outputs[-1].shape))
				targets[0].mult(cm.log(self.outputs[-1]),target = plus_loss)

				self.forget()

				#self.w2.subtract(e)
				#self.hidden_layer.i_IFOG.subtract(e)
				self.hidden_layer.hm1_IFOG.subtract(e)

				self.forward(x)

				minus_loss = cm.CUDAMatrix(np.zeros(self.outputs[-1].shape))
				targets[0].mult(cm.log(self.outputs[-1]),target = minus_loss)

				grads = plus_loss.subtract(minus_loss).mult(1./float(2*e))

				print('Grads',grads.asarray().argmax())
				print('gval', gvals[3].argmax())

				if seq % batch_size == 0:
					#print('Outputs:',enc.inverse_transform(self.outputs[-2].asarray()),enc.inverse_transform(self.outputs[-1].asarray()),'Input',enc.inverse_transform(x[-1][0].asarray()),'Target',enc.inverse_transform(targets[-1].asarray()))
					self.updateWeights()
					#self.lr = self.lr * decay
				self.reset_activations()
			
			print('Trained Epoch:',epoch+1,"With Accuracy:",acc)
		
class lstm_layer(object):
	def __init__(self, layers,uplim=1,lowlim=-1):
		super(lstm_layer, self).__init__()

		self.layers = layers

		#Input Gate Weights

		self.i_IFOG = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (self.layers[0],self.layers[1]*4)
		))

		self.hm1_IFOG = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (self.layers[1],self.layers[1]*4)
		))

		self.uplim = uplim
		self.lowlim = lowlim
		self.updates_tm1 = [cm.CUDAMatrix(np.zeros(self.i_IFOG.shape)),cm.CUDAMatrix(np.zeros(self.hm1_IFOG.shape))]
		self.forget()

	def forward(self,x):

		temp = cm.CUDAMatrix(np.zeros((1,self.layers[1])))

		gates = cm.dot(x,self.i_IFOG).add_dot(self.prev_outputs[-1],self.hm1_IFOG)
		self.prev_gates.append(gates)
		ifo = gates.get_col_slice(0,self.layers[1]*3)
		g = gates.get_col_slice(self.layers[1]*3,self.layers[1]*4)

		cm.sigmoid(ifo)
		cm.tanh(g)

		i = gates.get_col_slice(0,self.layers[1])
		f = gates.get_col_slice(self.layers[1],self.layers[1]*2)
		o = gates.get_col_slice(self.layers[1]*2,self.layers[1]*3)
		
		states = cm.CUDAMatrix(np.zeros((1,self.layers[1])))
		i.mult(g,target = states)
		f.mult(self.prev_states[-1],target = temp)
		states.add(temp)

		self.output = cm.CUDAMatrix(np.zeros((1,self.layers[1])))
		cm.tanh(states,target = self.output)
		self.output.mult(o)

		self.prev_outputs.append(self.output)
		self.prev_states.append(states)
		self.prev_fgates.append(gates)
		self.inputs.append(x)
		#print(self.output.asarray())
		return self.output

	def backward(self,grad,t):
		#print('prev_gc',self.prev_gc[-1].asarray())

		temp = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
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

		ec = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		ec.add(grad).add(recurrentGrad)

		#Loss wrt Cell State
		temp = cm.CUDAMatrix(np.ones(s.shape))
		es = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		temp.subtract(cm.pow(cm.tanh(s),2),target=es)
		#print(es.asarray())
		es.mult(fo).mult(ec)
		ff_tp1.mult(es_tp1,target=temp)
		es.add(temp)

		#Gradient at Output Gates
		go = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		cm.tanh(s,target = go)
		cl.mult_by_sigmoid_deriv(go,o)
		go.mult(ec)

		#Gradient at Cell Input
		temp = cm.CUDAMatrix(np.ones(s.shape))
		gg = temp.subtract(cm.pow(cm.tanh(g),2))
		fi.mult(es,target=gg)
		cl.mult_by_sigmoid_deriv(gg,g)

		#Gradient at Forget Gate
		gf = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		s_tm1.mult(es,target=gf)
		cl.mult_by_sigmoid_deriv(gf,f)

		#Gradient at Input Gate
		gi = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		cm.tanh(g,target=gi)
		gi.mult(es)
		cl.mult_by_sigmoid_deriv(gi,i)

		#self.clip(es)
		self.prev_es.append(es)

		ggates = cm.CUDAMatrix(np.zeros((1,self.layers[1]*4)))
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
		#print(self.gi_IFOG.asarray())
		#print(self.ghm1_IFOG.asarray())

	def clip(self,param):
		norm = param.euclid_norm()
		if norm > self.uplim:
			param.mult(float(self.uplim) / norm)

	def updateWeights(self,lr):
		#self.clip(self.ghm1_IFOG)
		print(self.i_IFOG.subtract(self.gi_IFOG.mult(lr).add(self.updates_tm1[0].mult(0.9))).asarray())
		print(self.hm1_IFOG.subtract(self.ghm1_IFOG.mult(lr).add(self.updates_tm1[1].mult(0.9))).asarray())
		self.updates_tm1 = [self.gi_IFOG,self.ghm1_IFOG]
		#print(self.i_IFOG.asarray())

	def forget(self):
		self.reset_activations()
		self.reset_grads()

	def reset_activations(self):
		#print('RESETTING')
		self.prev_states = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_outputs =[cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gates =[cm.CUDAMatrix(np.zeros([1,self.layers[1]*4]))]
		self.prev_fgates =[cm.CUDAMatrix(np.zeros([1,self.layers[1]*4]))]
		self.prev_ggates =[cm.CUDAMatrix(np.zeros([1,self.layers[1]*4]))]
		self.inputs = [cm.CUDAMatrix(np.zeros([1,self.layers[0]]))]
		self.gradInput = cm.CUDAMatrix(np.zeros([1,self.layers[0]]))

		self.prev_es = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))] 

	def reset_grads(self):
		#print('Resetting Grads')
		self.gi_IFOG = cm.CUDAMatrix(np.zeros((self.layers[0],self.layers[1]*4)))
		self.ghm1_IFOG = cm.CUDAMatrix(np.zeros((self.layers[1],self.layers[1]*4)))
		
		

ds = []
print('Loading Text')
with open('../data/ptb.train.short.txt') as doc:
	f = doc.read()
	words = f.split(' ')
	sequences = f.split('\n')
#print(text)
print('Building Dataset')
enc = prepro.LabelBinarizer()
enc.fit(words)
for x in sequences:
	w = x.split(' ')
	del w[-1]
	del w[0]
	seq = []
	for z in range(len(w)-1):
		i = cm.CUDAMatrix(enc.transform([w[z]]))
 		t = cm.CUDAMatrix(enc.transform([w[z+1]]))
 		seq.append([i,t])
 	ds.append(seq)

#ds = np.array(ds.tolist())
#print(ds.shape)
#print(ds[0][0][1])

n_tokens = enc.classes_.shape[0]
net = lstm([n_tokens,700,n_tokens])

start = timeit.timeit()
print('Starting Training')
net.train(ds,5,enc)
print('Time:',start)

net.last_best()

net.forget()
sent = [enc.inverse_transform(ds[0][12][0].asarray())]
for i in range(30):
	x = cm.CUDAMatrix(enc.transform([sent[-1]]))
	y = net.forward(x)
	sent.append(enc.inverse_transform(y.asarray())[0])

print(sent)

