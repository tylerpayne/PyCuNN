from numbapro import vectorize, cuda, float32
from numbapro.cudalib import cublas
import numpy as np
from timeit import default_timer as timer
import math
import gc

blas = cublas.Blas()
gc.collect()

def init_weights(n,gpu=True):
	w=None
	if gpu is True:
		w = np.array(np.random.uniform(
			low=max(-np.sqrt(6. / (n[0] + n[1])),-1),
			high=min(np.sqrt(6. / (n[0] + n[1])),1),
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

def load_sentences_data(fname,gpu=False):
	print('Building Dataset')
	global vocab
	vocab = {}
	global inv_vocab
	inv_vocab = []
	global word_idx
	word_idx = 0
	global total
	with open(fname,'r+') as doc:
		f = doc.read()
		sentences = f.split('\n')
		del sentences[-1]
		words = f.split(' ')
		total = len(words)

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
			if gpu == True:
				w = encode(w)
			sent.append(w)
		ds.append(sent)
	print('Dataset is Ready')
	return ds

def encode(word):
	if isinstance(word,basestring):
		x = np.zeros((1,word_idx),dtype='float32')
		x[0][vocab[word]] = 1.
		return cuda.to_device(x)
	else:
		return word
	

def decode(arr):
	if not isinstance(arr,basestring):
		a = np.zeros((arr.shape),dtype='float32')
		arr.copy_to_host(a)
		index = a.argmax(axis=1)
		return inv_vocab[index]
	else:
		return arr
	


#############


##SIGMOID

@cuda.jit('void(float32[:,:],float32[:,:],int32,int32)')
def d_msigmoid(a,b,dimx,dimy):
    x,y = cuda.grid(2)
    if (x<dimx) and (y<dimy):
        b[x,y] = 1. / (1. + math.exp(-a[x,y]))

@cuda.jit('float32(float32)',device=True)
def d_sigmoid(a):
    return 1. / (1. + math.exp(-a))

def msigmoid(a,b):
	assert a.shape == b.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],b.shape[0],b.shape[1])
	blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))
	d_msigmoid[gridDim,blockDim](a,b,a.shape[0],a.shape[1])

##SIGMOID DERIV

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],int32,int32)')
def d_msigmoid_deriv(a,h,b,dimx,dimy):
    x,y = cuda.grid(2)
    if (x<dimx) and (y<dimy):
        b[x,y] = ((1. - h[x,y])*h[x,y])*a[x,y]

def msigmoid_deriv(a,h,b):
	assert a.shape == b.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],b.shape[0],b.shape[1])
	blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_msigmoid_deriv[gridDim,blockDim](a,h,b,a.shape[0],a.shape[1])

##TANH_DERIV

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],int32,int32)')
def d_mtanh_deriv(a,h,b,dimx,dimy):
    x,y = cuda.grid(2)
    if (x<dimx) and (y<dimy):
        b[x,y] = (1. - (h[x,y]*h[x,y]))*a[x,y]

def mtanh_deriv(a,h,b):
	assert a.shape == b.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],b.shape[0],b.shape[1])
	blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_mtanh_deriv[gridDim,blockDim](a,h,b,a.shape[0],a.shape[1])

##TANH

@cuda.jit('void(float32[:,:],float32[:,:],int32,int32)')
def d_mtanh(a,b,dimx,dimy):
    x,y = cuda.grid(2)
    if (x<dimx) and (y<dimy):
        b[x,y] = math.tanh(a[x,y])

def mtanh(a,b):
	assert a.shape == b.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],b.shape[0],b.shape[1])
	blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_mtanh[gridDim,blockDim](a,b,a.shape[0],a.shape[1])

## SOFTMAX

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],int32)')
def d_msoftmax(a,s,b,dim):
    y = cuda.grid(1)
    if (y<dim):
        b[0,y] = math.exp(a[0,y]) / s[0,0]

def msoftmax(a,b,stream=None):
	assert a.shape == b.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],b.shape[0],b.shape[1])
	blockDim = (min(30,a.shape[1]))
	gridDim = (((a.shape[1] + blockDim) - 1) / blockDim)

	s = mexpsum(a)

	d_msoftmax[gridDim,blockDim,stream](a,s,b,a.shape[1])

#IFOG ACTIVATE

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:])')
def d_ifog_activate(i,f,o,g):
    x,y = cuda.grid(2)
    if (x<i.shape[0] and y < i.shape[1]):
        	i[x,y] = d_sigmoid(i[x,y])
        	f[x,y] = d_sigmoid(f[x,y])
        	o[x,y] = d_sigmoid(o[x,y])
        	g[x,y] = math.tanh(g[x,y])

def ifog_activate(gates):
    blockDim = (min(30,gates[0].shape[0]),min(30,gates[0].shape[1]))
    gridDim = ((((gates[0].shape[0] + blockDim[0]) - 1) / blockDim[0]), (((gates[0].shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_ifog_activate[gridDim,blockDim](gates[0],gates[1],gates[2],gates[3])

#IFOG BUILD
@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:])')
def d_ifog_build(ifog,i,f,o,g):
    x,y = cuda.grid(2)
    if (x<i.shape[0] and y < i.shape[1]):
        	ifog[x,y] = i[x,y]
        	ifog[x,y+i.shape[1]] = f[x,y]
        	ifog[x,y+(i.shape[1]*2)] = o[x,y]
        	ifog[x,y+(i.shape[1]*3)] = g[x,y]

def ifog_build(ifog,gates):
    blockDim = (min(30,gates[0].shape[0]),min(30,gates[0].shape[1]))
    gridDim = ((((gates[0].shape[0] + blockDim[0]) - 1) / blockDim[0]), (((gates[0].shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_ifog_build[gridDim,blockDim](ifog,gates[0],gates[1],gates[2],gates[3])

#IFOG SPLIT

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:],float32[:,:],float32[:,:])')
def d_ifog_split(ifog,i,f,o,g):
    x,y = cuda.grid(2)
    if (x<ifog.shape[0]):
    	if (y<i.shape[1]):
        	i[x,y] = ifog[x,y]
        if(y<g.shape[1]*2 and y>=i.shape[1]):
        	f[x,y] = ifog[x,y]
        if(y<f.shape[1]*3 and y >=g.shape[1]*2):
        	o[x,y] = ifog[x,y]
        if(y<(ifog.shape[1]) and y >= f.shape[1]*3):
        	g[x,y] = ifog[x,y]

def ifog_split(a,arr):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_ifog_split[gridDim,blockDim](a,arr[0],arr[1],arr[2],arr[3])



#FP

def fp(x,W,b,r):
   	mmprod(x,W,r)
   	mmadd(r,b,r)

#BP

def bp(grad,W,gW,gb,h,gradInput=None):
	z = cuda.device_array_like(gW)
	mmprod(h,grad,z,transa='T')
	mmadd(gW,z,gW)
	mmadd(gb,grad,gb)
	if gradInput is not None:
		mmprod(grad,W,gradInput,transb='T')

def accum_bp(grad,gW,gb,h):
	z = cuda.device_array_like(gW)
	mmprod(h,grad,z,transa='T')
	mmadd(gW,z,gW)
	mmadd(gb,grad,gb)

#DOT PRODUCT

def mmprod(a,b,c,transa='N',transb='N'):
	m = a.shape[0]
	n = b.shape[1]
	k = a.shape[1]
	if transa == 'T':
		m = a.shape[1]
		k=a.shape[0]
	if transb == 'T':
		n = b.shape[0]

	blas.gemm(transa,transb,m,n,k,1.0,a,b,0.0,c)

#ADD/SUBTRACT

def mmadd(a,b,c):
	blas.geam('N','N',a.shape[0],a.shape[1],1.0,a,1.0,b,c)

def mmsubtract(a,b,c):
	blas.geam('N','N',a.shape[0],a.shape[1],1.0,a,-1.0,b,c)

#SUM

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_msum(a,b):
	sA = cuda.shared.array(shape=(1024),dtype=float32)
	idx = cuda.threadIdx.x
	gidx = cuda.grid(1)
	total = min(cuda.blockDim.x,a.shape[1] - (cuda.blockIdx.x*cuda.blockDim.x))
	s = total/2
	if gidx+s < a.shape[1]:
		sA[idx] = a[0,gidx] + a[0,gidx+s]
		cuda.syncthreads()
		if total%2 == 1:
				if idx == total-1:
					sA[0] += a[0,idx] 
		s = s/2
		while s > 0:
			if idx < s:
				sA[idx] += sA[idx + s]
			cuda.syncthreads()
			if s%2 == 1 and s > 1:
				if idx == s-1:
					sA[0] += sA[idx] 
			cuda.syncthreads()
			s = s/2
		if idx == 0:
			b[0,cuda.blockIdx.x] = sA[0]

def msum(a):
	blockDim = min(1024,a.shape[1])
	gridDim = (((a.shape[1]) + blockDim) - 1) / blockDim
	db = cuda.device_array_like(a)
	d_msum[gridDim,blockDim](a,db)
	while gridDim > 1:
		last_gridDim = gridDim
		blockDim = gridDim
		gridDim = ((last_gridDim + blockDim) - 1) / blockDim
		d_msum[gridDim,blockDim](db,db)
	return db

#EXPSUM

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mexpsum(a,b):
	sA = cuda.shared.array(shape=(1024),dtype=float32)
	idx = cuda.threadIdx.x
	gidx = cuda.grid(1)
	total = min(cuda.blockDim.x,a.shape[1] - (cuda.blockIdx.x*cuda.blockDim.x))
	s = total/2
	if gidx+s < a.shape[1]:
		sA[idx] = math.exp(a[0,gidx]) + math.exp(a[0,gidx+s])
		cuda.syncthreads()
		if total%2 == 1:
				if idx == total-1:
					sA[0] += math.exp(a[0,idx])
		s = s/2
		while s > 0:
			if idx < s:
				sA[idx] += sA[idx + s]
			cuda.syncthreads()
			if s%2 == 1 and s > 1:
				if idx == s-1:
					sA[0] += sA[idx] 
			cuda.syncthreads()
			s = s/2
		if idx == 0:
			b[0,cuda.blockIdx.x] = sA[0]

def mexpsum(a):
	blockDim = min(1024,a.shape[1])
	gridDim = (((a.shape[1]) + blockDim) - 1) / blockDim
	db = cuda.device_array_like(a)
	d_mexpsum[gridDim,blockDim](a,db)
	while gridDim > 1:
		last_gridDim = gridDim
		blockDim = gridDim
		gridDim = ((last_gridDim + blockDim) - 1) / blockDim
		d_msum[gridDim,blockDim](db,db)
	return db
	



#MULT

@cuda.jit('void(float32[:,:],float32,float32[:,:])')
def d_msmult(a,b,c):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        c[x,y] = a[x,y]*b

def msmult(a,b,c):
	assert a.shape == c.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],c.shape[0],c.shape[1])
	blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_msmult[gridDim,blockDim](a,b,c)

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_mmmult(a,b,c):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        c[x,y] = a[x,y]*b[x,y]

def mmmult(a,b,c):
	assert a.shape == b.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],b.shape[0],b.shape[1])
	assert a.shape == c.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],c.shape[0],c.shape[1])
	blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_mmmult[gridDim,blockDim](a,b,c)

#TRANSPOSE

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mtranspose(a,b):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        b[y,x] = a[x,y]

def mtranspose(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    b = cuda.device_array((a.shape[1],a.shape[0]),dtype='float32')

    d_mtranspose[gridDim,blockDim](a,b)

    return b

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

    b = cuda.device_array_like(a)

    d_mcopy[gridDim,blockDim](a,b)

    return b

#CLIP

@cuda.jit('void(float32[:,:])')
def d_mclip(a):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        if (a[x,y] > 1.):
        	a[x,y] = 1.
        if (a[x,y] < -1.):
        	a[x,y] = -1.

def mclip(a):
    blockDim = (min(30,a.shape[0]),min(30,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mclip[gridDim,blockDim](a)

#UPDATE WEIGHTS

@cuda.jit('void(float32[:,:],float32[:,:],float32)')
def d_update_weights(w,gw,lr):
    x,y = cuda.grid(2)
    if (x < w.shape[0]) and (y <w.shape[1]):
        w[x,y] = w[x,y] - (lr*gw[x,y])

def update_weights(w,gw,lr):
	assert w.shape == gw.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(w.shape[0],w.shape[1],gw.shape[0],gw.shape[1])
	blockDim = (min(30,w.shape[0]),min(30,w.shape[1]))
	gridDim = ((((w.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((w.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_update_weights[gridDim,blockDim](w,gw,lr)

#ARGMAX

@cuda.jit('void(float32[:,:],int32[:,:],int32[:])')
def d_margmax(a,amax,aidx):
	sA = cuda.shared.array(shape=(1024),dtype=float32)
	sIdx = cuda.shared.array(shape=(1024),dtype=float32)
	idx = cuda.threadIdx.x
	gidx = cuda.grid(1)
	total = min(cuda.blockDim.x,a.shape[1] - (cuda.blockIdx.x*cuda.blockDim.x))
	s = total/2
	if gidx+s < a.shape[1]:
		if a[0,gidx] > a[0,gidx+s]:
			sA[idx] = a[0,gidx]
			sIdx[idx] = gidx
		else:
			sA[idx] = a[0,gidx+s]
			sIdx[idx] = gidx+s
		cuda.syncthreads()
		if total%2 == 1:
				if idx == total-1:
					if a[0,gidx] > a[0,gidx-idx]:
						sA[0] = a[0,gidx]
						sIdx[0] = gidx
		s = s/2
		while s > 0:
			if idx < s:
				if a[0,gidx] > a[0,gidx+s]:
					sA[idx] = a[0,gidx]
					sIdx[idx] = gidx
				else:
					sA[idx] = a[0,gidx+s]
					sIdx[idx] = gidx+s
			cuda.syncthreads()
			if s%2 == 1 and s > 1:
				if idx == s-1:
					if a[0,gidx] > a[0,gidx-idx]:
						sA[0] = a[0,gidx]
						sIdx[0] = gidx
			cuda.syncthreads()
			s = s/2
		if idx == 0:
			amax[0,cuda.blockIdx.x] = sA[0]
			aidx[0] = sIdx[0]

def margmax(a):
	blockDim = min(1024,a.shape[1])
	gridDim = (((a.shape[1]) + blockDim) - 1) / blockDim
	amax = cuda.device_array((1,gridDim),dtype='float32')
	aidx = cuda.device_array((gridDim),dtype='int32')
	d_margmax[gridDim,blockDim](a,amax,aidx)
	while gridDim > 1:
		print(amax[2])
		last_gridDim = gridDim
		blockDim = gridDim
		gridDim = ((last_gridDim + blockDim) - 1) / blockDim
		bmax = cuda.device_array((1,gridDim),dtype='float32')
		bidx = cuda.device_array((gridDim),dtype='int32')
		d_margmax[gridDim,blockDim](amax,bmax,bidx)
	return aidx




