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
		for _ in range(t.shape[0]-1):
			self.outputs[-1].subtract(t[-1],target=self.gOutput)
			self.gw2.add_dot(self.hidden_layer.prev_outputs[-1].T,self.gOutput)
			self.gb2.add_sums(self.gOutput,axis=0)

			self.delta = cm.dot(self.gOutput,self.w2.T)

			self.hidden_layer.backward(self.delta)

			self.outputs.pop()
			self.hs.pop()
			self.inputs.pop()
			t = np.delete(t,t.shape[0]-1)

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
				x = ds_x[seq*seq_len:(seq+1)*seq_len]
				d = ds_t[seq*seq_len:(seq+1)*seq_len]
				for t in range(x.shape[0]):
					self.forward(x[t])
				self.bptt(d)
				if seq % batch_size == 0:
					print('Output:',enc.inverse_transform(self.outputs[-1].asarray()),'Input',enc.inverse_transform(x[-1].asarray()),'Target',enc.inverse_transform(d[-1].asarray()))
					self.updateWeights()
					self.lr = self.lr * decay
				self.reset_activations()


	def reset_grads(self):
		self.gw1 = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))
		self.gb1 = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.gOutput = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.delta = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))

	def reset_activations(self):
		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.outputs = []
		self.h = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hs = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.inputs=[]
		self.hidden_layer.reset_activations()

	def forget(self):
		self.reset_grads()
		self.reset_activations()
		self.hidden_layer.forget()

		
class lstm_layer(object):
	def __init__(self, layers):
		super(lstm_layer, self).__init__()

		self.layers = layers

		self.i_ig_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
			size = (self.layers[0],self.layers[1])
		))

		self.hm1_ig_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.c_ig_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))


		#Forget Gate

		self.i_fg_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
			size = (self.layers[0],self.layers[1])
		))

		self.hm1_fg_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.c_fg_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (1,self.layers[1])
		))

		#Output Gate

		self.i_og_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
			size = (self.layers[0],self.layers[1])
		))

		self.hm1_og_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.c_og_weight = cm.CUDAMatrix(np.random.uniform(
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

		self.hm1_c_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[1])),
			size = (self.layers[1],self.layers[1])
		))

		self.forget()

	def forward(self,x):
		temp = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		#self.prev_states[-1].mult(self.c_ig_weight,target=temp)
		ai = cm.dot(x,self.i_ig_weight).add_dot(self.prev_outputs[-1],self.hm1_ig_weight)#.add(temp)
		bi = cm.sigmoid(ai)

		temp = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		#self.prev_states[-1].mult(self.c_fg_weight,target=temp)
		af = cm.dot(x,self.i_fg_weight).add_dot(self.prev_outputs[-1],self.hm1_fg_weight)#.add(temp)
		bf = cm.sigmoid(af)

		ac = cm.dot(x,self.i_c_weight).add_dot(self.prev_outputs[-1],self.hm1_c_weight)
		sc = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))

		temp = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		bf.mult(self.prev_states[-1],target=temp)
		sc.add(temp)
		temp = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		bi.mult(cm.sigmoid(ac),target=temp)
		sc.add(temp)

		temp = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		#self.prev_states[-1].mult(self.c_og_weight,target=temp)
		ao = cm.dot(x,self.i_og_weight).add_dot(self.prev_outputs[-1],self.hm1_og_weight)#.add(temp)
		bo = cm.sigmoid(ao)

		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		bo.mult(cm.sigmoid(sc),target=self.output)

		self.prev_states.append(sc)
		self.prev_outputs.append(self.output)
		self.inputs.append(x)
		self.prev_ac.append(ac)
		self.prev_ao.append(ao)
		self.prev_bf.append(bf)
		self.prev_af.append(af)
		self.prev_ai.append(ai)
		self.prev_bi.append(bi)
		self.prev_bo.append(bo)

		return self.output

	def backward(self,grad):
		inSum = cm.dot(self.prev_gc[-1],self.hm1_c_weight.T).add_dot(self.prev_gi[-1],self.hm1_ig_weight.T).add_dot(self.prev_gf[-1],self.hm1_fg_weight.T).add_dot(self.prev_go[-1],self.hm1_og_weight.T)
		inSum.add_dot(self.prev_gradInput[-1],self.i_c_weight).add_dot(self.prev_gradInput[-1],self.i_ig_weight).add_dot(self.prev_gradInput[-1],self.i_fg_weight).add_dot(self.prev_gradInput[-1],self.i_og_weight)
		ec = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		ec.add(grad).add(inSum)

		go = cm.CUDAMatrix(np.ones([1,self.layers[1]])) 
		cl.mult_by_sigmoid_deriv(go,self.prev_ao[-1])
		go.mult(cm.sigmoid(self.prev_states[-1]).mult(ec))

		es = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		cl.mult_by_sigmoid_deriv(es,self.prev_states[-1])
		es.mult(self.prev_bo[-1]).mult(ec).add(self.prev_bf[-1].mult(self.prev_es[-1])).add(self.prev_gi[-1].mult(self.c_ig_weight)).add(self.prev_gf[-1].mult(self.c_fg_weight)).add(go.mult(self.c_og_weight))

		gc = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		cl.mult_by_sigmoid_deriv(gc,self.prev_ac[-1])
		gc.mult(self.prev_bi[-1]).mult(es)

		gf = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		cl.mult_by_sigmoid_deriv(gf,self.prev_af[-2])
		gf.mult(self.prev_states[-2].mult(es))

		gi = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		cl.mult_by_sigmoid_deriv(gi,self.prev_ai[-1])
		gi.mult(cm.sigmoid(self.prev_ac[-1]).mult(es))

		#Accumulate Gradients

		gradInput = cm.dot(self.prev_gc[-1],self.i_c_weight.T).add_dot(self.prev_gi[-1],self.i_ig_weight.T).add_dot(self.prev_gf[-1],self.i_fg_weight.T).add_dot(self.prev_go[-1],self.i_og_weight.T)
		
		self.i_c_gweight.add_dot(self.inputs[-1].T,gc)
		self.hm1_c_gweight.add_dot(self.prev_outputs[-1].T,gc)

		self.i_ig_gweight.add_dot(self.inputs[-1].T,gi)
		self.hm1_ig_gweight.add_dot(self.prev_outputs[-1].T,gi)

		self.i_fg_gweight.add_dot(self.inputs[-1].T,gf)
		self.hm1_fg_gweight.add_dot(self.prev_outputs[-1].T,gf)

		self.i_og_gweight.add_dot(self.inputs[-1].T,go)
		self.hm1_og_gweight.add_dot(self.prev_outputs[-1].T,go)

		#temp = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		#temp.mult(gi).mult(self.prev_states[-1])
		#self.c_ig_gweight.add(temp)

		#temp = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		#temp.mult(gf).mult(self.prev_states[-1])
		#self.c_fg_gweight.add(temp)

		#temp = cm.CUDAMatrix(np.ones([1,self.layers[1]]))
		#temp.mult(go).mult(self.prev_states[-1])
		#self.c_og_gweight.add(temp)


		self.prev_ec.append(ec)
		self.prev_es.append(es)
		self.prev_go.append(go)
		self.prev_gc.append(gc)
		self.prev_gf.append(gf)
		self.prev_gi.append(gi)
		self.prev_gradInput.append(gradInput)
		self.prev_outputs.pop()
		self.inputs.pop()
		self.prev_states.pop()


	def updateWeights(self, lr):
		self.i_og_weight.subtract(self.i_og_gweight.mult(lr))
		self.hm1_og_weight.subtract(self.hm1_og_gweight.mult(lr))
		#self.c_og_weight.subtract(self.c_og_gweight.mult(lr))

		self.i_fg_weight.subtract(self.i_fg_gweight.mult(lr))
		self.hm1_fg_weight.subtract(self.hm1_fg_gweight.mult(lr))
		#self.c_fg_weight.subtract(self.c_fg_gweight.mult(lr))

		self.i_ig_weight.subtract(self.i_ig_gweight.mult(lr))
		self.hm1_ig_weight.subtract(self.hm1_ig_gweight.mult(lr))
		#self.c_ig_weight.subtract(self.c_ig_gweight.mult(lr))

		self.i_c_weight.subtract(self.i_c_gweight.mult(lr))
		self.hm1_c_weight.subtract(self.hm1_c_gweight.mult(lr))		

	def forget(self):
		self.reset_activations()
		self.reset_grads()

	def reset_activations(self):
		self.prev_states = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_outputs =[cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.inputs = []

		self.prev_ac = []
		self.prev_bo = []
		self.prev_ao = []
		self.prev_bf = []
		self.prev_af = []
		self.prev_ai = []
		self.prev_bi = []

		self.prev_ec = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_es = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_go = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gc = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gf = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gi = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_gradInput = [cm.CUDAMatrix(np.zeros([1,self.layers[0]]))]

	def reset_grads(self):

		self.hm1_c_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_c_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.c_og_gweight = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hm1_og_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_og_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.c_fg_gweight = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hm1_fg_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_fg_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.c_ig_gweight = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hm1_ig_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.i_ig_gweight = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		

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
net = lstm([n_tokens,1000,n_tokens])

start = timeit.timeit()
print('Starting Training')
net.train(ds,3,enc)
print('Time:',start)

net.forget()
seq = [enc.inverse_transform(ds[0][13][0].asarray())]
for i in range(30):
	x = cm.CUDAMatrix(enc.transform([seq[-1]]))
	y = net.forward(x)
	seq.append(enc.inverse_transform(y.asarray())[0])

print(seq)


