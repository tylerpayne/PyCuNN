import numpy as np 
import cudamat as cm 
from cudamat import learn as cl

cm.cublas_init()

class nn(object):
	def __init__(self, layers):
		super(nn, self).__init__()
		self.layers = layers
		self.output = None

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

		self.gw1 = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))

		self.gb1 = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))

	def forward(self,x):
		self.input = x
		self.h = cm.sigmoid(cm.dot(x,self.w1).add(self.b1))
		logits = cm.exp(cm.dot(self.h,self.w2).add(self.b2))
		self.output = logits.mult_by_col(cm.pow(cm.sum(logits,axis=1),-1))
		return self.output

	def backward(self,t):

		assert t.shape[0] is self.input.shape[0], "Ensure Same Batch Size as Forward Pass"

		self.delta = cm.CUDAMatrix(np.zeros([t.shape[0],self.layers[1]]))
		self.gw1 = cm.CUDAMatrix(np.zeros([self.layers[0],self.layers[1]]))
		self.gw2 = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))
		self.gb1 = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.gb2 = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		gOutput = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))

		self.output.subtract(t,target=gOutput)
		self.gw2.add_dot(self.h.T,gOutput)
		self.gb2.add_sums(gOutput,axis = 0)
		
		self.delta = cm.dot(gOutput,self.w2.T)
		cl.mult_by_sigmoid_deriv(self.delta,self.h)

		self.gw1.add_dot(self.input.T,self.delta)
		self.gb1.add_sums(self.delta,axis = 0)

		self.w2.subtract(self.gw2.mult(0.01))
		self.b2.subtract(self.gb2.mult(0.01))
		self.w1.subtract(self.gw1.mult(0.01))
		self.b1.subtract(self.gb1.mult(0.01))

	def train(self,ds_x,ds_t,batch_size=1,epochs):

		assert ds_x.shape[0] is ds_t.shape[0], "Size Mismatch: Ensure number of examples in input and target datasets is equal"

		for epoch in range(epochs):
			print('Epoch:',epoch)
			for batch in range(len(ds)/batch_size):
				x = ds_x[batch*batch_size:(batch+1)*batch_size]
				t = ds_t[batch*batch_size:(batch+1)*batch_size]

				self.forward(x)
				self.backward(t)


			




