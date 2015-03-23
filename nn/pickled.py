import pickle
import multiprocessing as mp
import numpy as np 
import utils
from sklearn import preprocessing as pp 
from timeit import default_timer as timer

def encode_sentences(enc,sentences,s):
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
		for sent in seq:
			ds.append(sent)
	print('Built Dataset')
	f = open('../data/ptb.pickle/ptbpickle%i.pickle' % s, 'w')
	pickle.dump(ds,f)
	f.close()
	print('wrote file')
	s += 1
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


with open('../data/ptb.train.txt','r+') as doc:
	f = doc.read()
	words = f.split(' ')
	sentences = f.split('\n')

enc = pp.LabelBinarizer()
enc.fit(words)

i = 0
o = len(sentences)/200

for s in range(199):
	ds = encode_sentences(enc,sentences[i:o],s)
	ds = None
	i = o
	o = o+len(sentences)/200

ds = utils.encode_sentences(enc,sentences[o:])

f = open('../data/ptb.pickle/encoder', 'w')
pickle.dump(enc,f)
f.close()
