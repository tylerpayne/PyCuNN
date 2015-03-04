import cudamat as cm 
from cudamat import learn as cl
import numpy as np
import copy

cm.cublas_init()

class lstm(object):
	def __init__(self, layers):
		super(lstm, self).__init__()

		assert len(layers) is 3, "Only one-hidden-layer LSTM Netowrks are supported at this time"
		
		self.layers = layers
		self.outputs = []
		self.step = 0

		# Build Netowrk

		#LSTM Layer

		self.hidden_layer = lstm_layer(layers)

		#Hidden to Output Weights

		self.h_y_weight = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (self.layers[1],self.layers[2])
		))

		self.h_y_bias = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (self.layers[1] + self.layers[2])),
			high=np.sqrt(6. / (self.layers[1] + self.layers[2])),
			size = (1,self.layers[2])
		))

		self.reset_grads()

	def forward(self,x):
		logits = cm.exp(cm.dot(self.hidden_layer.forward(x),self.h_y_weight).add(self.h_y_bias))
		output = logits.mult_by_col(cm.pow(cm.sum(logits,axis=1),-1)) 
		self.step = self.step + 1
		self.outputs.insert(0,output)
		return output

	def backward(self,target):
		self.reset_grads()
		self.outputs[0].subtract(target)
		self.h_y_gweight.add_dot(self.hidden_layer.output.T,self.outputs[0])
		self.h_y_gbias.subtract(self.outputs[0]) 

		#self.hidden_layer.backward(cm.dot(self.outputs[0],self.h_y_weight.T))

		self.h_y_weight.subtract(self.h_y_gweight.mult(0.01)) 
		self.h_y_bias.subtract(self.h_y_gbias.mult(0.01))

		#LSTM
		#rnn['hidden'].output = 

	def forget(self):
		self.step = 0
		self.outputs = []
		self.hidden_layer.forget()
		self.reset_grads()

	def reset_grads(self):
		self.h_y_gbias = cm.CUDAMatrix(np.zeros([1,self.layers[2]]))
		self.h_y_gweight = cm.CUDAMatrix(np.zeros([self.layers[1],self.layers[2]]))

		
class lstm_layer(object):
	def __init__(self, layers):
		super(lstm_layer, self).__init__()

		self.layers = layers
		self.output = None
		self.prev_states = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_outputs =[cm.CUDAMatrix(np.zeros([1,self.layers[1]]))] 
		self.states = []

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

		self.reset_grads()

		#Losses

		self.do = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.dc = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.df = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.di = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))

	def forward(self,x):
		self.ai = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		temp = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.prev_states[0].mult(self.c_ig_weight,target=temp)
		self.ai.add_dot(x,self.i_ig_weight).add_dot(self.prev_outputs[0],self.hm1_ig_weight).add(temp)
		self.bi = cm.sigmoid(self.ai)

		self.prev_states[0].mult(self.c_fg_weight,target=temp)
		self.af = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.af.add_dot(x,self.i_fg_weight).add_dot(self.prev_outputs[0],self.hm1_fg_weight).add(temp)
		self.bf = cm.sigmoid(self.af)

		self.ac = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.ac.add_dot(x,self.i_c_weight).add_dot(self.prev_outputs[0],self.hm1_c_weight)
		self.sc = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))

		self.bf.mult(self.prev_states[0],target=temp)
		self.sc.add(temp)
		self.bi.mult(self.ac,target=temp)
		self.sc.add(temp)

		self.prev_states[0].mult(self.c_og_weight,target=temp)
		self.ao = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.ao.add_dot(x,self.i_og_weight).add_dot(self.prev_outputs[0],self.hm1_og_weight).add(temp)
		self.bo = cm.sigmoid(self.ao)

		self.output = cm.CUDAMatrix(np.zeros([1,self.layers[1]]))
		self.bo.mult(cm.sigmoid(self.sc),target=temp)
		self.output.add(temp)

		self.prev_states.insert(0,self.sc)
		self.prev_outputs.insert(0,self.output)

		return self.output

	def backward(self,grad):
		self.prev_outputs[-1] = cm.CUDAMatrix(np.zeros(self.prev_outputs[-1].shape))
		self.prev_outputs[-1].add(grad).add(cm.dot(self.do,self.c_og_weight)).add(cm.dot(self.dc,self.i_c_weight)).add(cm.dot(self.df,self.c_fg_weight)).add(cm.dot(self.di,self.c_ig_weight))
		self.ao.mult(-1,target=self.do).add(1,target=self.do)
		self.do.mult(self.ao).mult(cm.sigmoid(self.prev_states[-1]).mult(self.prev_outputs[-1]))
		#self.

	def forget(self):
		self.prev_states = [cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.prev_outputs =[cm.CUDAMatrix(np.zeros([1,self.layers[1]]))]
		self.reset_grads()

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
		




		