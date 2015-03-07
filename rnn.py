import numpy as np 
import cudamat as cm 
from cudamat import learn as cl
import timeit
from sklearn import preprocessing as prepro

cm.cublas_init()

class rnn(object):
	def __init__(self, layers):
		super(rnn, self).__init__()
		self.layers = layers

		self.w1 = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[0] + self.layers[1])),
			high=np.sqrt(6. / (self.layers[0] + self.layers[1])),
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
		self.updates_tm1 = []
		self.lr = 0.01
		self.forget()

	def forward(self,x):
		self.h = cm.sigmoid(cm.dot(x,self.w1).add(self.b1).add_dot(self.hs[-1],self.wr).add(self.br))
		logits = cm.exp(cm.dot(self.h,self.w2).add(self.b2))
		self.output = logits.mult_by_col(cm.pow(cm.sum(logits,axis=1),-1))
		self.inputs.append(x)
		self.hs.append(self.h)
		self.outputs.append(self.output)
		return self.output

	def bptt(self,t):
		self.outputs[-1].subtract(t[-1],target=self.gOutput)
		self.gw2.add_dot(self.hs[-1].T,self.gOutput)
		self.gb2.add_sums(self.gOutput,axis = 0)
		
		self.delta = cm.dot(self.gOutput,self.w2.T)
		cl.mult_by_sigmoid_deriv(self.delta,self.hs[-1])

		self.gw1.add_dot(self.inputs[-1].T,self.delta)
		self.gb1.add_sums(self.delta,axis = 0)

		for q in range(5):
			self.gwr.add_dot(self.hs[-2].T,self.delta)
			self.gbr.add_sums(self.delta,axis=0)

			self.delta = cm.dot(self.delta,self.wr.T)
			self.outputs[-2].subtract(t[-2],target=self.gOutput)

			self.gw2.add_dot(self.hs[-2].T,self.gOutput)
			self.gb2.add_sums(self.gOutput,axis=0)

			self.delta.add_dot(self.gOutput,self.w2.T)
			cl.mult_by_sigmoid_deriv(self.delta,self.hs[-2])		

			self.gw1.add_dot(self.inputs[-2].T,self.delta)
			self.gb1.add_sums(self.delta,axis = 0)

			self.hs.pop()
			self.inputs.pop()
			self.outputs.pop()
			t = np.delete(t,t.shape[0]-1)

		#print(self.gwr.asarray())

	def updateWeights(self):
		self.w2.subtract(self.gw2.mult(self.lr).add(self.updates_tm1[0].mult(0.9)))
		self.b2.subtract(self.gb2.mult(self.lr).add(self.updates_tm1[1].mult(0.9)))
		self.w1.subtract(self.gw1.mult(self.lr).add(self.updates_tm1[2].mult(0.9)))
		self.b1.subtract(self.gb1.mult(self.lr).add(self.updates_tm1[3].mult(0.9)))
		self.wr.subtract(self.gwr.mult(self.lr).add(self.updates_tm1[4].mult(0.9)))
		self.br.subtract(self.gbr.mult(self.lr).add(self.updates_tm1[5].mult(0.9)))
		self.updates_tm1 = [self.gw2.divide(self.lr),self.gb2.divide(self.lr),self.gw1.divide(self.lr),self.gb1.divide(self.lr),self.gwr.divide(self.lr),self.gbr.divide(self.lr)]
		self.forget()

	def train(self,ds,epochs,enc,seq_len=10,batch_size=1,lr=0.05,decay=0.99):
		#assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"
		ds_x = ds[:,:,0][0]
		ds_t = ds[:,:,1][0]
		self.lr = lr
		err = []
		for epoch in range(epochs):
			print('Epoch:',epoch+1)
			seq_len = int(np.random.uniform(low=10,high=75,size=(1))[0])
			#print(seq_len)
			for seq in range(ds.shape[1]/seq_len):
				x = ds_x[seq*seq_len:(seq+1)*seq_len]
				d = ds_t[seq*seq_len:(seq+1)*seq_len]
				for t in range(x.shape[0]):
					self.forward(x[t])
				#print('Output:',enc.inverse_transform(self.outputs[-1].asarray()),'Input',enc.inverse_transform(x[-1].asarray()),'Target',enc.inverse_transform(d[-1].asarray()))
				self.bptt(d)
				if seq % batch_size == 0:
					self.updateWeights()
					#self.lr = self.lr * decay
				self.reset_activations()

	def reset_grads(self):
		self.gw1 = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))
		self.gwr = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[1]]))
		self.gb1 = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.gbr = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.gOutput = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.delta = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.updates_tm1 = [self.gw2,self.gb2,self.gw1,self.gb1,self.gwr,self.gbr]

	def reset_activations(self):
		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.outputs = []
		self.h = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.hs = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.inputs=[]

	def forget(self):
		self.reset_grads()
		self.reset_activations()

ds = []
print('Loading Text')
with open('./ptb.train.txt') as doc:
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
net = rnn([n_tokens,1000,n_tokens])

start = timeit.timeit()
print('Starting Training')
net.train(ds,30,enc)
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

