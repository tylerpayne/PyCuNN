from numbapro import vectorize, cuda
from numbapro.cudalib import cublas
from sklearn import preprocessing
import cudamat as cm
import numpy as np
import cProfile as profile
from timeit import default_timer as timer
import math

blas = cublas.Blas()
cm.cublas_shutdown()
cm.shutdown()
cm.init()
cm.cublas_init()

##SIGMOID

@cuda.jit('void(float32[:,:])')
def d_msigmoid(a):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        a[x,y] = 1. / (1. + math.exp(-a[x,y]))

@cuda.jit('float32(float32)',device=True)
def d_sigmoid(a):
    return 1. / (1. + math.exp(-a))

def msigmoid(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_msigmoid[gridDim,blockDim](a)

##SIGMOID DERIV

@cuda.jit('void(float32[:,:])')
def d_msigmoid_deriv(a):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        s = d_sigmoid(a[x,y])
        a[x,y] = (1. - s)*s

def msigmoid_deriv(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_msigmoid_deriv[gridDim,blockDim](a)

##TANH_DERIV

@cuda.jit('void(float32[:,:])')
def d_mtanh_deriv(a):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        a[x,y] = 1. - math.pow(math.tanh(a[x,y]),2)

def mtanh_deriv(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mtanh_deriv[gridDim,blockDim](a)

##TANH

@cuda.jit('void(float32[:,:])')
def d_mtanh(a):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        a[x,y] = math.tanh(a[x,y])

def mtanh(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mtanh[gridDim,blockDim](a)

## SOFTMAX

@cuda.jit('void(float32[:,:],float32[:],float32[:,:])')
def d_msoftmax(j,s,z):
    y = cuda.grid(1)

    if (y<j.shape[1]):
        e = math.exp(j[0,y])

    cuda.syncthreads()

    if (y<j.shape[1]):
        cuda.atomic.add(s,0,e)

    cuda.syncthreads()

    if (y<j.shape[1]):
        z[0,y] = e / s[0]

    cuda.syncthreads()

def msoftmax(j,z):
    blockDim = (min(30,j.shape[1]))
    gridDim = (((j.shape[1] + blockDim) - 1) / blockDim)

    print(gridDim,blockDim)

    s = np.zeros((1),dtype='float32')
    ds = cuda.to_device(s)

    d_msoftmax[gridDim,blockDim](j,ds,z)

    ds.copy_to_host(s)

    print(s)

#FP

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:])')
def d_fp(i,W,b,r):
    x,y = cuda.grid(2)
    equalDim = W.shape[0]
    temp = 0
    for z in range(equalDim):
        temp += i[x,z] * W[z,y]
    temp += b[x,y]
    cuda.syncthreads()
    cuda.atomic.add(r,(x,y),temp)

def fp(x,W,b,r):
    blockDim = (min(30,W.shape[0]),min(30,W.shape[1]))
    gridDim = ((((W.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((W.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_fp[gridDim,blockDim](x,W,b,r)


#DOT PRODUCT

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_mmprod(a,b,c):
    x,y = cuda.grid(2)
    if (x < c.shape[0]) and (y <c.shape[1]):
        equalDim = a.shape[1]
        temp = 0
        for z in range(equalDim):
            temp += a[x,z] * b[z,y]
        cuda.syncthreads()
        cuda.atomic.add(c,(x,y),temp)

def mmprod(a,b,c):
    blockDim = (min(30,a.shape[0]),min(30,b.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((b.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mmprod[gridDim,blockDim](a,b,c)

#ADD/SUBTRACT

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_mmadd(a,b,c):
    x,y = cuda.grid(2)
    if (x < c.shape[0]) and (y <c.shape[1]):
        c[x,y] = a[x,y] + b[x,y]

def mmadd(a,b,c):
    blockDim = (min(30,a.shape[0]),min(30,b.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((b.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mmadd[gridDim,blockDim](a,b,c)

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_mmsubtract(a,b,c):
    x,y = cuda.grid(2)
    if (x < c.shape[0]) and (y <c.shape[1]):
        c[x,y] = a[x,y] - b[x,y]

def mmsubtract(a,b,c):
    blockDim = (min(30,a.shape[0]),min(30,b.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((b.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mmsubtract[gridDim,blockDim](a,b,c)



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
    a = np.array(np.random.rand(3000,6000),dtype='float32')
    c = np.array(np.random.rand(6000,200),dtype='float32')
    o = np.zeros((3000,200),dtype='float32')
    da = cuda.to_device(a)
    dc = cuda.to_device(c)
    do = cuda.to_device(o)

    start = timer()
    mmprod(da,dc,do)
    msigmoid(do)
    msigmoid(do)
    cutime = timer()-start
    print('cutime',cutime)

    b = cm.CUDAMatrix(a)
    d = cm.CUDAMatrix(c)

    start = timer()
    g = cm.sigmoid(cm.sigmoid(cm.dot(b,d)))
    cmtime = timer() -start
    print('cmtime',cmtime)

    print('perf imporvement',cmtime/cutime)

    print('cmoutput',g.asarray())

    do.copy_to_host(o)

    print('cuoutput',o)
    
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














