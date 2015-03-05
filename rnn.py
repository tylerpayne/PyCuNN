import numpy as np 
import cudamat as cm 
from cudamat import learn as cl

cm.cublas_init()

class rnn(object):
	def __init__(self, layers):
		super(rnn, self).__init__()
		self.layers = layers

		self.w1 = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (self.layers[0],self.layers[1])
		))

		self.b1 = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (1,self.layers[1])
		))

		self.w2 = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (self.layers[1],self.layers[2])
		))

		self.b2 = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (1,self.layers[2])
		))

		self.wr = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (self.layers[1],self.layers[1])
		))

		self.br = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (1,self.layers[1])
		))

		self.inputs = []

		self.forget()

	def forward(self,x):
		self.inputs.append(x)
		self.h = cm.sigmoid(cm.dot(x,self.w1).add(self.b1).add_dot(self.hs[-1],self.wr).add(self.br))
		logits = cm.exp(cm.dot(self.h,self.w2).add(self.b2))
		self.output = logits.mult_by_col(cm.pow(cm.sum(logits,axis=1),-1))
		self.hs.append(self.h)
		self.outputs.append(self.output)
		return self.output

	def bptt(self,t):
		
		self.outputs[-1].subtract(t,target=self.gOutput)
		self.gw2.add_dot(self.hs[-1].T,self.gOutput)
		self.gb2.add_sums(self.gOutput,axis = 0)
		
		self.delta = cm.dot(self.gOutput,self.w2.T)
		cl.mult_by_sigmoid_deriv(self.delta,self.hs[-1])

		for _ in range(5):
			self.deltas.append(self.delta)
			self.gwr.add_dot(self.hs[-2].T,self.deltas[-1])
			self.gbr.add_sums(self.delta,axis=0)

			self.gw1.add_dot(self.inputs[-1].T,self.delta)
			self.gb1.add_sums(self.delta,axis = 0)

			self.delta = cm.dot(self.delta,self.wr.T)

			self.hs.pop()
			self.inputs.pop()

	def updateWeights(self):
		self.w2.subtract(self.gw2.mult(0.01))
		self.b2.subtract(self.gb2.mult(0.01))
		self.w1.subtract(self.gw1.mult(0.01))
		self.b1.subtract(self.gb1.mult(0.01))
		self.wr.subtract(self.gwr.mult(0.01))
		self.br.subtract(self.gbr.mult(0.01))
		self.forget()

	def train(self,ds,epochs,batch_size=None):

		if batch_size == None:
			batch_size = ds.shape[1]
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		ds_x = ds[:,:,0][0]
		ds_t = ds[:,:,1][0]
		for epoch in range(epochs):
			print('Epoch:',epoch+1)
			for batch in range(ds.shape[1]/batch_size):
				x = ds_x[batch*batch_size:(batch+1)*batch_size]
				y = ds_t[(batch+1)*batch_size]
				for t in range(x.shape[0]):
					self.forward(x[t])
				self.bptt(y)
				self.updateWeights()
				

	def reset_grads(self):
		self.gw1 = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))
		self.gwr = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.gb1 = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.gbr = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.gOutput = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.delta = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))

	def forget(self):
		self.reset_grads()
		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.outputs = []
		self.h = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hs = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.deltas = []
		self.inputs=[]


import timeit
from sklearn import preprocessing as prepro


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
net = rnn([n_tokens,800,n_tokens])

start = timeit.timeit()
print('Starting Training')
net.train(ds,10,batch_size=5)
print('Time:',start)

net.forget()
y = net.forward(ds[0][0][0])
seq = [y]

for i in range(30):
	seq.append(net.forward(seq[-1]))

sent = []
for x in seq:
	sent.append(enc.inverse_transform(x.asarray()))

print(sent)

			




