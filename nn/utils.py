from numbapro import vectorize, cuda
from numbapro.cudalib import cublas
import cudamat as cm
import numpy as np
from timeit import default_timer as timer
import math

blas = cublas.Blas()
cm.cublas_shutdown()
cm.shutdown()
cm.init()
cm.cublas_init()

def init_weights(n,gpu=True):
	w=None
	if gpu is True:
		w = np.array(np.random.uniform(
			low=-np.sqrt(6. / (n[0] + n[1])),
			high=np.sqrt(6. / (n[0] + n[1])),
			size = (n[0],n[1])
		),dtype='float32')
		w = cuda.to_device(w)
	else:
		w = np.random.uniform(
			low=-np.sqrt(6. / (n[0] + n[1])),
			high=np.sqrt(6. / (n[0] + n[1])),
			size = (n[0],n[1])
		)
	return(w)

def zeros(n,gpu=True):
	w=None
	if gpu is True:
		w = np.zeros((n[0],n[1]),dtype='float32')
		w = cuda.to_device(w)
	else:
		w = np.zeros((n[0],n[1]))
	return(w)

def asarray(a):
	x = np.zeros((a.shape),dtype='float32')
	a.copy_to_host(x)
	return x

def load_data(fname):
	global vocab
	vocab = {}
	global inv_vocab
	inv_vocab = []
	global word_idx
	word_idx = 0
	with open(fname,'r+') as doc:
		f = doc.read()
		sentences = f.split('\n')
		del sentences[-1]
		words = f.split(' ')

	for i in range(len(words)-1):
		if words[i] not in vocab:
			vocab[words[i]] = word_idx
			inv_vocab.append(words[i])
			word_idx += 1
	ds = []
	for seq in sentences:
		seq = seq.split(' ')
		del seq[0]
		del seq[-1]
		sent = []
		for w in seq:
			sent.append(w)
		ds.append(sent)
	return ds

def encode(word):
	x = np.zeros((1,word_idx))
	x[0][vocab[word]] = 1.
	return cuda.to_device(x)

def decode(arr):
	a = np.zeros((arr.shape))
	arr.copy_to_host(a)
	index = a.argmax(axis=1)
	return inv_vocab[index]


#############


##SIGMOID

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_msigmoid(a,b):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        b[x,y] = 1. / (1. + math.exp(-a[x,y]))

@cuda.jit('float32(float32)',device=True)
def d_sigmoid(a):
    return 1. / (1. + math.exp(-a))

def msigmoid(a,b):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_msigmoid[gridDim,blockDim](a,b)

##SIGMOID DERIV

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_msigmoid_deriv(a,b):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        b[x,y] = (1. - a[x,y])*a[x,y]

def msigmoid_deriv(a,b):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_msigmoid_deriv[gridDim,blockDim](a,b)


##TANH_DERIV

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mtanh_deriv(a,b):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        b[x,y] = 1. - math.pow(math.tanh(a[x,y]),2)

def mtanh_deriv(a,b):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mtanh_deriv[gridDim,blockDim](a,b)

##TANH

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mtanh(a,b):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        b[x,y] = math.tanh(a[x,y])

def mtanh(a,b):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mtanh[gridDim,blockDim](a,b)

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

#IFOG ACTIVATE

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:])')
def d_ifog_activate(ifog,i,f,o,g):
    x,y = cuda.grid(2)
    if (x<ifog.shape[0]):
    	if (y<i.shape[1]):
        	i[x,y] = d_sigmoid(ifog[x,y])
        elif(y<g.shape[1]*2):
        	f[x,y] = d_sigmoid(ifog[x,y])
        elif(y<f.shape[1]*3):
        	o[x,y] = d_sigmoid(ifog[x,y])
        elif(y<(ifog.shape[1])):
        	g[x,y] = math.tanh(ifog[x,y])

def ifog_activate(ifog,gates):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_ifog_activate[gridDim,blockDim](ifog,gates[0],gates[1],gates[2],gates[3])

#IFOG CONSTRUCT
@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:])')
def d_ifog_build(ifog,i,f,o,g):
    x,y = cuda.grid(2)
    if (x<ifog.shape[0]):
    	if (y<i.shape[1]):
        	ifog[x,y] = i[x,y]
        elif(y<g.shape[1]*2):
        	ifog[x,y] = f[x,y]
        elif(y<f.shape[1]*3):
        	ifog[x,y] = o[x,y]
        elif(y<(ifog.shape[1])):
        	ifog[x,y] = g[x,y]

def ifog_build(ifog,gates):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_ifog_activate[gridDim,blockDim](ifog,gates[0],gates[1],gates[2],gates[3])


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

#BP

def bp(grad,W,gW,gb,h):
	mmprod(mtranspose(h),grad,gW)
	mmadd(gb,grad,gb)
	gradInput = zeros([1,W.shape[0]])
	mmprod(grad,mtranspose(W),gradInput)
	mclip(gradInput)
	mclip(gb)
	mclip(gW)
	return gradInput


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

#MULT

@cuda.jit('void(float32[:,:],float32,float32[:,:])')
def d_msmult(a,b,c):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        c[x,y] = a[x,y]*b

def msmult(a,b,c):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_msmult[gridDim,blockDim](a,b,c)

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_mmmult(a,b,c):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        c[x,y] = a[x,y]*b[x,y]

def mmmult(a,b,c):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_msmult[gridDim,blockDim](a,b,c)

#TRANSPOSE

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mtranspose(a,b):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        b[y,x] = a[x,y]

def mtranspose(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    b = np.zeros((a.shape),dtype='float32')
    db = cuda.to_device(b)

    d_mtranspose[gridDim,blockDim](a,b)

    return db

#ZERO

@cuda.jit('void(float32[:,:])')
def d_mzero(a):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        a[x,y] = 0.

def mzero(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mzero[gridDim,blockDim](a)

#Copy

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mcopy(a,b):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        b[x,y] = a[x,y]

def mcopy(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    b = np.zeros((a.shape),dtype='float32')
    db = cuda.to_device(b)

    d_mcopy[gridDim,blockDim](a,db)

    return db

#IFOG SPLIT

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:])')
def d_ifog_split(ifog,i,f,o,g):
    x,y = cuda.grid(2)
    if (x<ifog.shape[0]):
    	if (y<i.shape[1]):
        	i[x,y] = ifog[x,y]
        elif(y<g.shape[1]*2):
        	f[x,y] = ifog[x,y]
        elif(y<f.shape[1]*3):
        	o[x,y] = ifog[x,y]
        elif(y<(ifog.shape[1])):
        	g[x,y] = ifog[x,y]

def ifog_split(a,arr):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_ifog_split[gridDim,blockDim](a,arr[0],arr[1],arr[2],arr[3])

#CLIP

@cuda.jit('void(float32[:,:])')
def d_mclip(a):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        if (a[x,y] > 15.):
        	a[x,y] = 15.;
        if (a[x,y] < -15.):
        	a[x,y] = -15.;

def mclip(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mclip[gridDim,blockDim](a)




