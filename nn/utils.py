import numpy as np 
import cudamat as cm 
from numbapro import vectorize,cuda
import multiprocessing as mp
from sklearn import preprocessing as pp
from timeit import default_timer as timer


'''
class LabelBinarizer(object):
	def __init__(self, pos_label = 1,neg_label = 0):
		super(LabelBinarizer, self).__init__()
		self.pos_label = pos_label
		self.neg_label = neg_label

	def fit(self,textArray):
		self.vocab = np.array([],dtype='|S5')
		dvocab = cuda.to_device(self.vocab)
		dtext = cuda.to_device(textArray)

	@cuda.jit('void(char[:],char')
	def iterate(arr,test,ret):
'''

def init_weights(n_in,n_out,gpu=True):
	w=None
	if gpu is True:
		w = cm.CUDAMatrix(np.random.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size = (n_in,n_out)
		))
	else:
		w = np.random.uniform(
			low=-np.sqrt(6. / (n_in + n_out)),
			high=np.sqrt(6. / (n_in + n_out)),
			size = (n_in,n_out)
		)
	return(w)

def zeros(n_in,n_out,gpu=True):
	w=None
	if gpu is True:
		w = cm.CUDAMatrix(np.zeros((n_in,n_out)))
	else:
		w = np.zeros((n_in,n_out))
	return(w)

def encode_sentences(enc,sentences):
	num_processes = 8
	z,x = 0,len(sentences)/num_processes
	processes = []
	output = mp.Queue()
	for _ in range(num_processes-1):
		processes.append(mp.Process(target=encode,args=(enc,sentences[z:x],output)))
		z = x
		x = x+len(sentences)/num_processes
	processes.append(mp.Process(target=encode,args=(enc,sentences[z:],output)))
	for p in processes:
		p.start()
	print('Building Dataset')
	results = [output.get() for p in processes]
	for p in processes:
		p.join()
	ds = []
	for seq in results:
		print(len(seq))
		for sent in seq:
			ds.append(sent)
	print('Built Dataset')
	return ds

def encode(enc,sentences,output):
	p = mp.current_process()
	seq = []
	for sent in sentences:
		s = []
		for z in range(len(sent)-2):
			i = enc.transform([sent[z]])
 			t = enc.transform([sent[z+1]])
 			s.append([i,t])
 		seq.append(s)
 	output.put(seq)


