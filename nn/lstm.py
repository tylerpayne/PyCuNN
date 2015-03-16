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
	def __init__(self, layers,uplim=8,lowlim=-8):
		super(lstm, self).__init__()

		assert len(layers) is 3, "Only one-hidden-layer LSTM Netowrks are supported at this time"
		
		self.layers = layers
		self.outputs = []
		self.uplim = uplim
		self.lowlim = lowlim

		# Build Netowrk

		#LSTM Layer

		self.hidden_layer = lstm_layer(layers,uplim,lowlim)

		#Hidden to Output Weights

		self.w2 = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (self.layers[1],self.layers[2])
		))

		self.b2= cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (1,self.layers[2])
		))

		self.forget()
		self.updates_tm1 = [self.gw2,self.gb2]

	def forward(self,x):
		self.h = self.hidden_layer.forward(x)
		logits = cm.exp(cm.dot(self.h,self.w2).add(self.b2))
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
	
		#print(t.shape[0],len(self.outputs),len(self.inputs),len(self.hidden_layer.prev_outputs),len(self.hidden_layer.prev_states),len(self.hidden_layer.prev_ac))
		
		for _ in range(t.shape[0]-1,-1,-1):	
			#print('Delta',self.delta.asarray())
			self.outputs[_+1].subtract(t[_],target=self.gOutput)
			self.clip(self.gw2.add_dot(self.hidden_layer.prev_outputs[_+1].T,self.gOutput))
			self.clip(self.gb2.add_sums(self.gOutput,axis=0))

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


	def train(self,ds,epochs,enc,seq_len=45,batch_size=1,lr=0.05,decay=0.99):
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		ds_x = ds[:,:,0][0]
		ds_t = ds[:,:,1][0]
		self.lr = lr/batch_size
		
		for epoch in range(epochs):
			correct = 0
			print('Begin Epoch:',epoch+1)
			seq_len = int(np.random.uniform(low=50,high=175,size=(1))[0])
			#print(seq_len)
			print(ds.shape[1])
			for seq in range(ds.shape[1]/seq_len):
				#print('seq',seq)
				x = ds_x[seq*seq_len:(seq+1)*seq_len]
				d = ds_t[seq*seq_len:(seq+1)*seq_len]
				for t in range(x.shape[0]):
					self.forward(x[t])
					if d[t].argmax(axis=1).asarray()[0][0] == self.outputs[-1].argmax(axis=1).asarray()[0][0]:
						correct += 1
				self.bptt(d)
				if seq % batch_size == 0:
					#print('Outputs:',enc.inverse_transform(self.outputs[-2].asarray()),enc.inverse_transform(self.outputs[-1].asarray()),'Input',enc.inverse_transform(x[-1].asarray()),'Target',enc.inverse_transform(d[-1].asarray()))
					self.updateWeights()
					#self.lr = self.lr * decay
				self.reset_activations()
			print('Trained Epoch:',epoch+1,"Accuracy:",float(correct)/float(ds.shape[1]))


	def reset_grads(self):
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.gOutput = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.delta = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))

	def reset_activations(self):
		#print("MAIN RESET")
		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.outputs = [cm.CUDAMatrix(np.zeros([1,self.layers[2]]))]
		self.h = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hs = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.inputs=[cm.CUDAMatrix(np.zeros([1,self.layers[0]]))]
		self.hidden_layer.reset_activations()

	def forget(self):
		self.reset_grads()
		self.reset_activations()
		self.hidden_layer.forget()

		
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
		
		gates = cm.dot(x,self.i_IFOG).add(cm.dot(self.prev_outputs[-1],self.hm1_IFOG))
		self.prev_gates.append(gates)
		cm.sigmoid(gates)

		i = gates.get_col_slice(0,self.layers[1])
		f = gates.get_col_slice(self.layers[1],self.layers[1]*2)
		o = gates.get_col_slice(self.layers[1]*2,self.layers[1]*3)
		g = gates.get_col_slice(self.layers[1]*3,self.layers[1]*4)

		states = cm.CUDAMatrix(np.zeros((1,self.layers[1])))
		i.mult(g,target = states)
		f.mult(self.prev_states[-1],target=temp)
		states.add(temp)

		self.output = cm.CUDAMatrix(np.zeros((1,self.layers[1])))
		cm.sigmoid(states,target=self.output)
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
		es = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		fo.mult(ec,target = es)
		cl.mult_by_sigmoid_deriv(es,s)
		ff_tp1.mult(es_tp1,target=temp)
		es.add(temp)

		#Gradient at Output Gates
		go = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		cm.sigmoid(s,target = go)
		cl.mult_by_sigmoid_deriv(go,o)
		go.mult(ec)

		#Gradient at Cell Input
		gg = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		fi.mult(es,target=gg)
		cl.mult_by_sigmoid_deriv(gg,g)

		#Gradient at Forget Gate
		gf = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		s_tm1.mult(es,target=gf)
		cl.mult_by_sigmoid_deriv(gf,f)

		#Gradient at Input Gate
		gi = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		cm.sigmoid(g,target=gi)
		gi.mult(es)
		cl.mult_by_sigmoid_deriv(gi,i)

		self.clip(es)
		self.prev_es.append(es)

		ggates = cm.CUDAMatrix(np.zeros((1,self.layers[1]*4)))
		ggates.set_col_slice(0,self.layers[1],gi)
		ggates.set_col_slice(self.layers[1],self.layers[1]*2,gf)
		ggates.set_col_slice(self.layers[1]*2,self.layers[1]*3,go)
		ggates.set_col_slice(self.layers[1]*3,self.layers[1]*4,gg)

		self.clip(ggates)
		self.prev_ggates.append(ggates)

		#Accumulate Gradients

		di = self.gi_IFOG.get_col_slice(0,self.layers[1])
		df = self.gi_IFOG.get_col_slice(self.layers[1],self.layers[1]*2)
		do = self.gi_IFOG.get_col_slice(self.layers[1]*2,self.layers[1]*3)
		dg = self.gi_IFOG.get_col_slice(self.layers[1]*3,self.layers[1]*4)

		di.add_dot(self.inputs[t].T,gi)
		df.add_dot(self.inputs[t].T,gf)
		do.add_dot(self.inputs[t].T,go)

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
		self.prev_states = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_outputs =[cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gates =[cm.CUDAMatrix(np.zeros([1,self.layers[1]*4]))]
		self.prev_fgates =[cm.CUDAMatrix(np.zeros([1,self.layers[1]*4]))]
		self.prev_ggates =[cm.CUDAMatrix(np.zeros([1,self.layers[1]*4]))]
		self.inputs = [cm.CUDAMatrix(np.zeros([1,self.layers[0]]))]

		self.prev_es = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))] 
		#self.prev_gradInput = [cm.CUDAMatrix(np.zeros([1,self.layers[0]]))]

	def reset_grads(self):
		#print('Resetting Grads')
		self.gi_IFOG = cm.CUDAMatrix(np.zeros((self.layers[0],self.layers[1]*4)))
		self.ghm1_IFOG = cm.CUDAMatrix(np.zeros((self.layers[1],self.layers[1]*4)))
		
		

ds = []
print('Loading Text')
with open('../data/ptb.train.short.txt') as doc:
	text = doc.read().rstrip("\n").split(" ")
print('Building Dataset')
enc = prepro.LabelBinarizer()
enc.fit(text)
for x in range(len(text)-1):
	i = cm.CUDAMatrix(enc.transform([text[x]]))
 	t = cm.CUDAMatrix(enc.transform([text[x+1]]))
 	ds.append([i,t])

ds = np.array([ds])

n_tokens = enc.classes_.shape[0]
net = lstm([n_tokens,800,n_tokens])

start = timeit.timeit()
print('Starting Training')
net.train(ds,10,enc)
print('Time:',start)

net.forget()
seq = [enc.inverse_transform(ds[0][12][0].asarray())]
for i in range(30):
	x = cm.CUDAMatrix(enc.transform([seq[-1]]))
	y = net.forward(x)
	seq.append(enc.inverse_transform(y.asarray())[0])

print(seq)

