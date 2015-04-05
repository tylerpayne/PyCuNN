import numpy as np 
import utils
from utils import *
from numbapro import cuda
from timeit import default_timer as timer

class nn(object):
	def __init__(self, layers):
		super(nn, self).__init__()
		self.layers = layers
		self.output = None

		self.w1 = init_weights([self.layers[0],self.layers[1]])

		self.b1 = init_weights([1,self.layers[1]])

		self.w2 = init_weights([self.layers[1],self.layers[2]])

		self.b2 = init_weights([1,self.layers[2]])

		self.h = zeros([1,self.layers[1]])
		self.y = zeros([1,self.layers[2]])
		self.output = zeros([1,self.layers[2]])
		self.delta = zeros([1,self.layers[1]])
		self.gw1 = zeros([self.layers[0],self.layers[1]])
		self.gw2 = zeros([self.layers[1],self.layers[2]])
		self.gb1 = zeros([1,self.layers[1]])
		self.gb2 = zeros([1,self.layers[2]])
		self.gOutput = zeros([1,self.layers[2]])
		self.gInput = zeros([1,self.layers[0]])
		self.input = zeros([1,self.layers[0]])

	def forward(self,x):
		mzero(self.input)
		mzero(self.h)
		mzero(self.y)
		mzero(self.output)

		self.input = mcopy(x)
		fp(x,self.w1,self.b1,self.h)
		mtanh(self.h,self.h)
		fp(self.h,self.w2,self.b2,self.y)
		mtanh(self.y,self.output)
		return self.output

	def backward(self,t):
		mzero(self.gOutput)
		mzero(self.gInput)
		mzero(self.delta)
		mzero(self.gw2)
		mzero(self.gb2)
		mzero(self.gw1)
		mzero(self.gb1)

		mmsubtract(self.output,t,self.gOutput)
		bp(self.gOutput,self.w2,self.gw2,self.gb2,self.h,self.delta)
		mtanh_deriv(self.delta,self.h,self.delta)
		bp(self.delta,self.w1,self.gw1,self.gb1,self.input,self.gInput)

		update_weights(self.w1,self.gw1,0.01)
		update_weights(self.b1,self.gb1,0.01)
		update_weights(self.w2,self.gw2,0.01)
		update_weights(self.b2,self.gb2,0.01)

		#g = np.zeros(self.w1.shape,dtype='float32')
		#self.w1.copy_to_host(g)
		#print(g)

	def train(self,ds,epochs,batch_size=1):

		for epoch in range(epochs):
			start = timer()
			for i in range(len(ds)):
				x = ds[i][0]
				t = ds[i][1]
				assert x.shape[1] == self.layers[0]
				assert t.shape[1] == self.layers[2]
				#f = np.array([[0,0]],dtype='float32')
				self.forward(x)#.copy_to_host(f)
				#print('target',ds[i][1],'output',f)
				self.backward(t)
			print("Epoch",epoch,"Time Per Example",(timer()-start)/float(len(ds)))

a = [cuda.to_device(np.array([[-1,-1]],dtype='float32')),cuda.to_device(np.array([[1,-1]],dtype='float32'))]
b = [cuda.to_device(np.array([[-1,1]],dtype='float32')),cuda.to_device(np.array([[-1,1]],dtype='float32'))]
c = [cuda.to_device(np.array([[1,-1]],dtype='float32')),cuda.to_device(np.array([[-1,1]],dtype='float32'))]
d = [cuda.to_device(np.array([[1,1]],dtype='float32')),cuda.to_device(np.array([[1,-1]],dtype='float32'))]

ds = []
for _ in range(100):
	ds.append(a)
	ds.append(b)
	ds.append(c)
	ds.append(d)

net = nn([2,400,2])

net.train(ds,25)

x = cuda.to_device(ds[0][0])

f = np.array([[0,0]],dtype='float32')


net.forward(x).copy_to_host(f)
print(f)

x = cuda.to_device(ds[1][0])

f = np.array([[0,0]],dtype='float32')


net.forward(x).copy_to_host(f)
print(f)


x = cuda.to_device(ds[2][0])

f = np.array([[0,0]],dtype='float32')


net.forward(x).copy_to_host(f)
print(f)

x = cuda.to_device(ds[3][0])

f = np.array([[0,0]],dtype='float32')


net.forward(x).copy_to_host(f)
print(f)


			




