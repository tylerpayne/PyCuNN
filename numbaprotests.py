from numbapro import vectorize, cuda
from numbapro.cudalib import cublas
from sklearn import preprocessing
import cudamat as cm
import numpy as np
import cProfile as profile
from timeit import default_timer as timer

blas = cublas.Blas()

#@vectorize(["float32(float32,float32)"],target='gpu')
def run():
    a = np.random.rand(1000,1500)
    b = np.random.rand(1500,1000)

    start = timer()
    c = np.dot(a,b)
    nptime = timer()-start
    print('nptime',nptime)

    x = np.array(np.random.rand(1000,1500),dtype='float32',order='F')
    y = np.array(np.random.rand(1500,1000),dtype='float32',order='F')
    z = np.zeros_like(y,order='F')

    dx = cuda.to_device(x)
    dy = cuda.to_device(y)
    dz = cuda.to_device(z)

    start = timer()
    blas.gemm('N','N',1000,1500,1000,1.0,x,y,1.0,z)
    cutime = timer()-start
    print('cutime',cutime)

    x = cm.CUDAMatrix(np.random.rand(1000,1500))
    y = cm.CUDAMatrix(np.random.rand(1000,1000))
    
    start = timer()
    cm.dot(x,y)
    cmtime = timer()-start
    print('cmtime',cmtime)

run()
'''

def iter():
    start = timer()
    with open('./data/ptb.train.short.txt') as doc:
        f = doc.read()
        words = f.split(' ')
        seq = f.split('\n')

   
    enc = prepro.LabelBinarizer()
    enc.fit(words)
    ds = []
    for x in sequences:
        w = x.split(' ')
        del w[-1]
        del w[0]
        seq = []
        for z in range(len(w)-1):
            i = cm.CUDAMatrix(enc.transform([w[z]]))
            t = cm.CUDAMatrix(enc.transform([w[z+1]]))
            seq.append([i,t])
        ds.append(seq)
    slurp = timer() - start
    print('slurptime:' slurp)

    start = timer()
    ds=[]
    with open('./data/ptb.train.short.txt','r+') as doc:
        for line in doc:

   
    enc = prepro.LabelBinarizer()
    enc.fit(words)
    ds = []
    for x in sequences:
        w = x.split(' ')
        del w[-1]
        del w[0]
        seq = []
        for z in range(len(w)-1):
            i = cm.CUDAMatrix(enc.transform([w[z]]))
            t = cm.CUDAMatrix(enc.transform([w[z+1]]))
            seq.append([i,t])
        ds.append(seq)
    slurp = timer() - start
    print('slurptime:' slurp)


'''














