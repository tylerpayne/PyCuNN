import cudamat as cm
from cudamat import learn as cl
import numpy as np
import copy
from sklearn import preprocessing as prepro
import timeit

cm.cublas_init()

class lstm(object):
	def __init__(self, layers):
		super(lstm, self).__init__()

		assert len(layers) is 3, "Only one-hidden-layer LSTM Netowrks are supported at this time"
		
		self.layers = layers
		self.outputs = []

		# Build Netowrk

		#LSTM Layer

		self.hidden_layer = lstm_layer(layers)

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

	def forward(self,x):
		self.h = self.hidden_layer.forward(x)
		logits = cm.exp(cm.dot(self.h,self.w2).add(self.b2))
		self.output = logits.mult_by_col(cm.pow(cm.sum(logits,axis=1),-1))
		self.inputs.append(x)
		self.hs.append(self.h)
		self.outputs.append(self.output)
		return self.output

	def bptt(self,t):
		#Set T+1 activations to 0
		print('bptt')
		self.hidden_layer.prev_outputs.append(cm.empty(self.hidden_layer.prev_outputs[-1].shape))
		self.hidden_layer.prev_states.append(cm.empty(self.hidden_layer.prev_states[-1].shape))
		self.hidden_layer.prev_ac.append(cm.empty(self.hidden_layer.prev_ac[-1].shape))
		self.hidden_layer.prev_ao.append(cm.empty(self.hidden_layer.prev_ao[-1].shape))
		self.hidden_layer.prev_af.append(cm.empty(self.hidden_layer.prev_af[-1].shape))
		self.hidden_layer.prev_ai.append(cm.empty(self.hidden_layer.prev_ai[-1].shape))
		self.hidden_layer.prev_bf.append(cm.empty(self.hidden_layer.prev_bf[-1].shape))
		self.hidden_layer.prev_bo.append(cm.empty(self.hidden_layer.prev_bo[-1].shape))
		self.hidden_layer.prev_bi.append(cm.empty(self.hidden_layer.prev_bi[-1].shape))
		print(t.shape[0],len(self.outputs),len(self.inputs),len(self.hidden_layer.prev_outputs),len(self.hidden_layer.prev_states),len(self.hidden_layer.prev_ac))
		
		for _ in range(t.shape[0]-1,-1,-1):
			self.outputs[_+1].subtract(t[_],target=self.gOutput)
			self.gw2.add_dot(self.hidden_layer.prev_outputs[_+1].T,self.gOutput)
			self.gb2.add_sums(self.gOutput,axis=0)

			self.delta = cm.dot(self.gOutput,self.w2.T)
			#print(self.delta.asarray())
			self.hidden_layer.backward(self.delta,_+1)

	def updateWeights(self):
		self.w2.subtract(self.gw2.mult(self.lr))
		self.b2.subtract(self.gb2.mult(self.lr))
		self.hidden_layer.updateWeights(self.lr)
		self.forget()

	def train(self,ds,epochs,enc,seq_len=45,batch_size=1,lr=0.01,decay=0.999):
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		ds_x = ds[:,:,0][0]
		ds_t = ds[:,:,1][0]
		self.lr = lr/batch_size
		err = []
		for epoch in range(epochs):
			print('Epoch:',epoch+1)
			seq_len = int(np.random.uniform(low=45,high=100,size=(1))[0])
			#print(seq_len)
			for seq in range(ds.shape[1]/seq_len):
				print('seq',seq)
				x = ds_x[seq*seq_len:(seq+1)*seq_len]
				d = ds_t[seq*seq_len:(seq+1)*seq_len]
				for t in range(x.shape[0]):
					self.forward(x[t])
				self.bptt(d)
				if seq % batch_size == 0:
					print('Output:',enc.inverse_transform(self.outputs[-1].asarray()),'Input',enc.inverse_transform(x[-1].asarray()),'Target',enc.inverse_transform(d[-1].asarray()))
					self.updateWeights()
					#self.lr = self.lr * decay
				self.reset_activations()


	def reset_grads(self):
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.gOutput = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.delta = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))

	def reset_activations(self):
		print("MAIN RESET")
		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.outputs = [cm.empty([1,self.layers[2]])]
		self.h = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hs = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.inputs=[cm.empty([1,self.layers[0]])]
		self.hidden_layer.reset_activations()

	def forget(self):
		self.reset_grads()
		self.reset_activations()
		self.hidden_layer.forget()

		
class lstm_layer(object):
	def __init__(self, layers):
		super(lstm_layer, self).__init__()

		self.layers = layers

		#Input Gate Weights

		self.i_ig_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
			size = (self.layers[0],self.layers[1])
		))

		self.i_ig_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		self.hm1_ig_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.hm1_ig_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		#Forget Gate Weights

		self.i_fg_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
			size = (self.layers[0],self.layers[1])
		))

		self.i_fg_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		self.hm1_fg_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.hm1_fg_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		#Output Gate Weights

		self.i_og_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
			size = (self.layers[0],self.layers[1])
		))

		self.i_og_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		self.hm1_og_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.hm1_og_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		#Cell Weights

		self.i_c_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
			size = (self.layers[0],self.layers[1])
		))

		self.i_c_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		self.hm1_c_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.hm1_c_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		self.forget()

	def forward(self,x):
		self.inputs.append(x)
		temp = cm.empty((1,self.layers[1]))
		#Input Gates
		ai = cm.dot(x,self.i_ig_weight).add(self.i_ig_bias).add_dot(self.prev_outputs[-1],self.hm1_ig_weight).add(self.hm1_ig_bias)
		#print(ai.asarray())
		bi = cm.empty(ai.shape)
		cm.sigmoid(ai,target = bi)

		#Forget Gates
		af = cm.dot(x,self.i_fg_weight).add(self.i_fg_bias).add_dot(self.prev_outputs[-1],self.hm1_fg_weight).add(self.hm1_fg_bias)
		bf = cm.empty(af.shape)
		cm.sigmoid(af,target= bf)

		#Cell States
		ac = cm.dot(x,self.i_c_weight).add(self.i_c_bias).add_dot(self.prev_outputs[-1],self.hm1_c_weight).add(self.hm1_c_bias)
		sc = cm.empty(ac.shape)
		bf.mult(self.prev_states[-1],target=temp)
		cm.sigmoid(ac,target=sc)
		sc.mult(bi)
		sc.add(temp)
		
		#Output Gates
		ao = cm.dot(x,self.i_og_weight).add(self.i_og_bias).add_dot(self.prev_outputs[-1],self.hm1_og_weight).add(self.hm1_og_bias)
		bo = cm.empty(ao.shape) 
		cm.sigmoid(ao,target=bo)

		#Block Outputs
		self.output = cm.empty((1,self.layers[1]))
		cm.sigmoid(sc,target = self.output)
		self.output.mult(bo)

		#Remember Parameters
		self.prev_states.append(sc)
		self.prev_outputs.append(self.output)
		
		self.prev_ac.append(ac)
		self.prev_ao.append(ao)
		self.prev_bf.append(bf)
		self.prev_af.append(af)
		self.prev_ai.append(ai)
		self.prev_bi.append(bi)
		self.prev_bo.append(bo)
		#print(self.output.asarray())
		return self.output

	def backward(self,grad,t):
		#print('prev_gc',self.prev_gc[-1].asarray())

		temp = cm.empty((1,self.layers[1]))
		#Backpropogate Gradients from Gates at t+1 through Recurrent Weights to Block Output
		recurrentGrad = cm.dot(self.prev_gc[-1],self.hm1_c_weight.T).add_dot(self.prev_gi[-1],self.hm1_ig_weight.T).add_dot(self.prev_gf[-1],self.hm1_fg_weight.T).add_dot(self.prev_go[-1],self.hm1_og_weight.T)
		#print(recurrentGrad.asarray())

		#Gradient at Outputs
		#print(grad.asarray())
		ec = cm.empty((1,self.layers[1]))
		print(ec.asarray())
		ec.add(grad).add(recurrentGrad)
		#print(ec.asarray())

		#Gradient at Output Gates
		go = cm.empty((1,self.layers[1]))
		#print(self.prev_ao[t].asarray())
		cm.sigmoid(self.prev_states[t],target = go)
		cl.mult_by_sigmoid_deriv(go,self.prev_ao[t])
		#print(go.asarray())
		go.mult(ec)
		#print(go.asarray())

		#Loss wrt Cell State
		es = cm.empty([1,self.layers[1]])
		self.prev_bo[t].mult(ec,target = es)
		#print(es.asarray())
		cl.mult_by_sigmoid_deriv(es,self.prev_states[t])
		#print(es.asarray())
		self.prev_bf[t+1].mult(self.prev_es[-1],target=temp)
		es.add(temp)
		#print(es.asarray())

		#Gradient at Cell Input
		gc = cm.empty([1,self.layers[1]])
		self.prev_bi[t].mult(es,target=gc)
		cl.mult_by_sigmoid_deriv(gc,self.prev_ac[t])
		#print(gc.asarray())

		#Gradient at Forget Gate
		gf = cm.empty([1,self.layers[1]])
		self.prev_states[t-1].mult(es,target=gf)
		cl.mult_by_sigmoid_deriv(gf,self.prev_af[t])
		#print(gf.asarray())

		#Gradient at Input Gate
		gi = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		cm.sigmoid(self.prev_ac[t],target=temp)
		es.mult(temp,target=gi)
		cl.mult_by_sigmoid_deriv(gi,self.prev_ai[t])

		self.prev_ec.append(ec)
		self.prev_es.append(es)
		self.prev_go.append(go)
		self.prev_gc.append(gc)
		self.prev_gf.append(gf)
		self.prev_gi.append(gi)
		#print(len(self.prev_gc))
		#Accumulate Gradients

		#gradInput = cm.dot(self.prev_gc[-1],self.i_c_weight.T).add_dot(self.prev_gi[-1],self.i_ig_weight.T).add_dot(self.prev_gf[-1],self.i_fg_weight.T).add_dot(self.prev_go[-1],self.i_og_weight.T)
		#print('Input',self.inputs[t].asarray())
		self.i_c_gweight.add_dot(self.inputs[t].T,gc)
		self.hm1_c_gweight.add_dot(self.prev_outputs[t].T,gc)
		self.i_c_gbias.add_sums(gc,axis=0)
		self.hm1_c_gbias.add_sums(gc,axis=0)

		self.i_ig_gweight.add_dot(self.inputs[t].T,gi)
		self.hm1_ig_gweight.add_dot(self.prev_outputs[t].T,gi)
		self.i_ig_gbias.add_sums(gi,axis=0)
		self.hm1_ig_gbias.add_sums(gi,axis=0)

		self.i_fg_gweight.add_dot(self.inputs[t].T,gf)
		self.hm1_fg_gweight.add_dot(self.prev_outputs[t].T,gf)
		self.i_fg_gbias.add_sums(gf,axis=0)
		self.hm1_fg_gbias.add_sums(gf,axis=0)

		self.i_og_gweight.add_dot(self.inputs[t].T,go)
		self.hm1_og_gweight.add_dot(self.prev_outputs[t].T,go)
		self.i_og_gbias.add_sums(go,axis=0)
		self.hm1_og_gbias.add_sums(go,axis=0)

	def updateWeights(self,lr):
		self.i_og_weight.subtract(self.i_og_gweight.mult(lr))
		self.hm1_og_weight.subtract(self.hm1_og_gweight.mult(lr))
		self.i_og_bias.subtract(self.i_og_gbias.mult(lr))
		self.hm1_og_bias.subtract(self.hm1_og_gbias.mult(lr))

		self.i_fg_weight.subtract(self.i_fg_gweight.mult(lr))
		self.hm1_fg_weight.subtract(self.hm1_fg_gweight.mult(lr))
		self.i_fg_bias.subtract(self.i_fg_gbias.mult(lr))
		self.hm1_fg_bias.subtract(self.hm1_fg_gbias.mult(lr))

		self.i_ig_weight.subtract(self.i_ig_gweight.mult(lr))
		self.hm1_ig_weight.subtract(self.hm1_ig_gweight.mult(lr))
		self.i_ig_bias.subtract(self.i_ig_gbias.mult(lr))
		self.hm1_ig_bias.subtract(self.hm1_ig_gbias.mult(lr))

		self.i_c_weight.subtract(self.i_c_gweight.mult(lr))
		self.hm1_c_weight.subtract(self.hm1_c_gweight.mult(lr))
		self.i_c_bias.subtract(self.i_c_gbias.mult(lr))
		self.hm1_c_bias.subtract(self.hm1_c_gbias.mult(lr))

	def forget(self):
		self.reset_activations()
		self.reset_grads()

	def reset_activations(self):
		print('RESETTING')
		self.prev_states = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_outputs =[cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.inputs = [cm.empty([1,self.layers[0]])]

		self.prev_ac = [cm.empty([1,self.layers[1]])]
		self.prev_bo = [cm.empty([1,self.layers[1]])]
		self.prev_ao = [cm.empty([1,self.layers[1]])]
		self.prev_bf = [cm.empty([1,self.layers[1]])]
		self.prev_af = [cm.empty([1,self.layers[1]])]
		self.prev_ai = [cm.empty([1,self.layers[1]])]
		self.prev_bi = [cm.empty([1,self.layers[1]])]

		self.prev_ec = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_es = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_go = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gc = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gf = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gi = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		#self.prev_gradInput = [cm.CUDAMatrix(np.zeros([1,self.layers[0]]))]

	def reset_grads(self):
		print('Resetting Grads')
		self.hm1_c_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_c_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.hm1_c_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.i_c_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		
		self.hm1_og_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_og_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.hm1_og_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.i_og_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		
		self.hm1_fg_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_fg_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.hm1_fg_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.i_fg_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		
		self.hm1_ig_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_ig_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.hm1_ig_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.i_ig_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		

ds = []
print('Loading Text')
with open('./siddhartha.txt') as doc:
	text = doc.read().split(" ")
print('Building Dataset')
enc = prepro.LabelBinarizer()
enc.fit(text)
for x in range(len(text)-1):
	i = cm.CUDAMatrix(enc.transform([text[x]]))
 	t = cm.CUDAMatrix(enc.transform([text[x+1]]))
 	ds.append([i,t])

ds = np.array([ds])

n_tokens = enc.classes_.shape[0]
net = lstm([n_tokens,200,n_tokens])

start = timeit.timeit()
print('Starting Training')
net.train(ds,2,enc)
print('Time:',start)

net.forget()
seq = [enc.inverse_transform(ds[0][12][0].asarray())]
for i in range(30):
	x = cm.CUDAMatrix(enc.transform([seq[-1]]))
	y = net.forward(x)
	seq.append(enc.inverse_transform(y.asarray())[0])

print(seq)

