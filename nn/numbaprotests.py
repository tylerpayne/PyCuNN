from numbapro import vectorize, cuda
from numbapro.cudalib import cublas
from sklearn import preprocessing
import cudamat as cm
import numpy as np
import cProfile as profile
from timeit import default_timer as timer
import math
from utils import *

blas = cublas.Blas()

def run():
    """
    a = np.ones((1000,500),dtype='float32')
    c = np.ones((1000,500),dtype='float32')
    #o = np.zeros((1000,400),dtype='float32')
    da = cuda.to_device(a)
    dc = cuda.to_device(c)
    
    mmadd(da,dc,da)

    da.copy_to_host(a)

    print(a)

    """
    gc.collect()

    a = np.array(np.random.rand(1,5000),dtype='float32')
    c = np.zeros((1,1),dtype='int32')
    z = np.zeros((200,400),dtype='float32')
    #o = np.zeros((3000,200),dtype='float32')
    da = cuda.to_device(a)
    dc = cuda.to_device(c)

    print(np.argmax(a,axis=1))

    start = timer()
    g = margmax(da)
    cutime = timer()-start
    print('cutime',cutime)

    g.copy_to_host(c)
    print('CU',c)

    '''b = cm.CUDAMatrix(a)
    d = cm.CUDAMatrix(c)

    start = timer()
    b.T
    cmtime = timer() -start
    print('cmtime',cmtime)

    print('perf imporvement',cmtime/cutime)

    #print('cmoutput',g.asarray())'''

    
   # print(np.sum(c,axis=1))
    
def tests():
    a = np.random.rand(300,500)
    b = np.random.rand(500,300)

    start = timer()
    c = np.dot(a,b)
    nptime = timer()-start
    print('nptime',nptime)

    x = np.array(np.random.rand(600,1500),dtype='float32',order='F')
    y = np.array(np.random.rand(1500,300),dtype='float32',order='F')
    z = np.zeros((1000,1000),order='F',dtype='float32')

    stream = cuda.stream()

    dx = cuda.to_device(x)
    dy = cuda.to_device(y)
    dz = cuda.to_device(z)

    start = timer()
    blas.gemm('N','N',1000,1500,1000,1.0,dx,dy,0.0,dz)
    cutime = timer()-start
    print('cutime',cutime)

    #dz.copy_to_host(z)
    print(dz[0])

    c = np.ones((1000,1000),order='F',dtype='float32')
    print(c.shape)
    dc = cuda.to_device(c)

   # blockDim = (256,256)
    #gridDim = (((1000 + blockDim[0]-1)/blockDim[0]),((1000 + blockDim[1]-1)/blockDim[1]))

    blockDim = (30,30)
    gridDim = ((((c.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((c.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    start = timer()
    mtanh[gridDim,blockDim,stream](dc)
    tantime = timer() - start
    print('tantime',tantime)

    dc.copy_to_host(c,stream=stream)
    stream.synchronize()
    print(c)

    y = cm.CUDAMatrix(np.ones((1000,1000)))

    start = timer()
    cm.tanh(y)
    cmtan = timer()-start
    print('cmtan',cmtan)

    x = cm.CUDAMatrix(np.random.rand(1000,1500))
    y = cm.CUDAMatrix(np.random.rand(1500,1000))

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














