from numbapro import cuda, float32
from numbapro.cudalib import cublas
import numpy as np
from timeit import default_timer as timer
import math
import gc
import pickle
import utils

blas = cublas.Blas()
gc.collect()

def set_vocab(voc,inv,vecs,d_vecs):
	global d_vectors
	global vocab
	global inv_vocab
	global vectors

	vocab = voc
	inv_vocab = inv
	vectors = vecs
	d_vectors = d_vecs

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
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
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
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
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
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
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
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_mtanh[gridDim,blockDim](a,b,a.shape[0],a.shape[1])

## SOFTMAX

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_msoftmax(a,s,b):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
        b[0,y] = math.exp(a[0,y]) / s[0,0]

def msoftmax(a,b,stream=None):
	assert a.shape == b.shape, "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],b.shape[0],b.shape[1])
	blockDim = (min(32,a.shape[1]))
	gridDim = (((a.shape[1] + blockDim) - 1) / blockDim)

	s = mexpsum(a)

	d_msoftmax[gridDim,blockDim,stream](a,s,b)

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
    blockDim = (min(32,gates[0].shape[0]),min(32,gates[0].shape[1]))
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
    blockDim = (min(32,gates[0].shape[0]),min(32,gates[0].shape[1]))
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
    blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_ifog_split[gridDim,blockDim](a,arr[0],arr[1],arr[2],arr[3])

#FP

def fp(x,W,b,r):
	if r.shape[0] is not x.shape[0]:
		r = utils.zeros([x.shape[0],r.shape[1]])
   	mmprod(x,W,r)
   	madd_col_vec(r,b,r)

#BP

def bp(grad,W,gW,gb,h,gradInput=None):
	z = cuda.device_array_like(gW)
	print('grad',grad.shape,'h',h.shape)
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

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_madd_col_vec(a,b,c):
    x,y = cuda.grid(2)
    if (x<a.shape[0]) and (y<a.shape[1]):
    	c[x,y] = a[x,y] + b[0,y]
    	
def madd_col_vec(a,b,c):
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_madd_col_vec[gridDim,blockDim](a,b,c)

#SUM

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_msum(a,b):
	sA = cuda.shared.array(shape=(32,32),dtype=float32)
	xidx,yidx = cuda.threadIdx.x,cuda.threadIdx.y
	x,y = cuda.grid(2)
	total = min(cuda.blockDim.y, a.shape[1] - (cuda.blockIdx.y*cuda.blockDim.y))
	s = total/2
	if y+s < a.shape[1]:
		if yidx<s:
			sA[xidx,yidx] = a[x,y] + a[x,y+s]
		cuda.syncthreads()
		if yidx == total-1 and not yidx < s:
			sA[xidx,0] += a[x,y+s]
		cuda.syncthreads()
		last_s = s
		s=s/2
		while s>0:
			if yidx<s:
				sA[xidx,yidx] += sA[xidx,yidx+s]
			cuda.syncthreads()
			if yidx == last_s-1 and not yidx < s:
				sA[xidx,0] += sA[xidx,yidx+s]
			cuda.syncthreads()
			s=s/2
		if yidx==0:
			b[x,cuda.blockIdx.y] = sA[xidx,yidx]

'''
@cuda.jit('void(float32[:,:],float32[:,:])')
def d_msum(a,b):
	sA = cuda.shared.array(shape=(32,32),dtype=float32)
	xidx,yidx = cuda.threadIdx.x,cuda.threadIdx.y
	x,y = cuda.grid(2)
	total = min(cuda.blockDim.y, a.shape[1] - (cuda.blockIdx.y*cuda.blockDim.y))
	s = total/2
	if y+s < a.shape[1]:
		if yidx<s:
			sA[xidx,yidx] = a[x,y] + a[x,y+s]
		cuda.syncthreads()
		if yidx==total-1 and not yidx < s:
			sA[xidx,0] += a[x,y]
		cuda.syncthreads()
		last_s = s
		s = s/2
		while s>0:
			if yidx < s:
				sA[xidx,yidx] += sA[xidx,yidx+s]
			cuda.syncthreads()
			if yidx==last_s-1 and not yidx < s:
				sA[xidx,0] += sA[xidx,yidx]
			cuda.syncthreads()
			last_s = s
			s=s/2
		cuda.syncthreads()
		if yidx == 0:
			b[x,cuda.blockIdx.y] = sA[xidx,yidx]

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_msum(a,b):
	sA = cuda.shared.array(shape=(100,100),dtype=float32)
	xidx,yidx = cuda.threadIdx.x,cuda.threadIdx.y
	x,y = cuda.grid(2)
	total = min(cuda.blockDim.y, a.shape[1] - (cuda.blockIdx.y*cuda.blockDim.y))
	s = total/2
	if y+s < a.shape[1]:
		if yidx<s:
			sA[xidx,yidx] = a[x,y] + a[x,y+s]
		cuda.syncthreads()
		if yidx==total-1 and not yidx < s:
			sA[xidx,0] += a[x,y]
		cuda.syncthreads()
		last_s = s
		s = s/2
		while s>0:
			if yidx < s:
				sA[xidx,yidx] += sA[xidx,yidx+s]
			cuda.syncthreads()
			if yidx==last_s-1 and not yidx < s:
				sA[xidx,0] += sA[xidx,yidx]
			cuda.syncthreads()
			last_s = s
			s=s/2
		cuda.syncthreads()
		if yidx == 0:
			b[x,cuda.blockIdx.y] = sA[xidx,yidx]'''

def msum(a):
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))
	db = cuda.device_array_like(a)
	print(blockDim,gridDim)
	d_msum[gridDim,blockDim](a,db)
	while gridDim[1] > 1:
		blockDim = (min(32,a.shape[0]),min(32,gridDim[1]))
		gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((gridDim[1] + blockDim[1]) - 1) / blockDim[1]))
		print(blockDim,gridDim)
		d_msum[gridDim,blockDim](db,db)
	return db

#EXPSUM

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mexpsum(a,b):
	sA = cuda.shared.array(shape=(100,100),dtype=float32)
	xidx,yidx = cuda.threadIdx.x,cuda.threadIdx.y
	x,y = cuda.grid(2)
	total = min(cuda.blockDim.y, a.shape[1] - (cuda.blockIdx.y*cuda.blockDim.y))
	s = total/2
	if yidx<s:
		sA[xidx,yidx] = math.exp(a[x,y]) + math.exp(a[x,y+s])
	elif yidx+s==total-1:
		cuda.syncthreads()
		sA[xidx,0] += math.exp(a[x,y+s])
	cuda.syncthreads()
	last_s = s
	s = s/2
	while s>0:
		if yidx < s:
			sA[xidx,yidx] += sA[xidx,yidx+s]
		elif yidx+s==last_s-1:
			cuda.syncthreads()
			sA[xidx,0] += sA[xidx,yidx+s]
		cuda.syncthreads()
		last_s = s
		s=s/2
	cuda.syncthreads()
	if yidx == 0:
		b[x,cuda.blockIdx.y] = sA[xidx,yidx]

def mexpsum(a):
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))
	db = cuda.device_array_like(a)
	print(blockDim,gridDim)
	d_mexpsum[gridDim,blockDim](a,db)
	while gridDim[1] > 1:
		blockDim = (min(32,a.shape[0]),min(32,gridDim[1]))
		gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((gridDim[1] + blockDim[1]) - 1) / blockDim[1]))
		print(blockDim,gridDim)
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
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
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
	blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
	gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_mmmult[gridDim,blockDim](a,b,c)

#TRANSPOSE

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mtranspose(a,b):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        b[y,x] = a[x,y]

def mtranspose(a):
    blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
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
    blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    d_mzero[gridDim,blockDim](a)

#Copy

@cuda.jit('void(float32[:,:],float32[:,:])')
def d_mcopy(a,b):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        b[x,y] = a[x,y]

def mcopy(a):
    blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
    gridDim = ((((a.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((a.shape[1] + blockDim[1]) - 1) / blockDim[1]))

    b = cuda.device_array_like(a)

    d_mcopy[gridDim,blockDim](a,b)

    return b

#CLIP

@cuda.jit('void(float32[:,:])')
def d_mclip(a):
    x,y = cuda.grid(2)
    if (x < a.shape[0]) and (y <a.shape[1]):
        if (a[x,y] > 0.9):
        	a[x,y] = 0.9
        if (a[x,y] < -0.9):
        	a[x,y] = -0.9

def mclip(a):
    blockDim = (min(32,a.shape[0]),min(32,a.shape[1]))
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
	blockDim = (min(32,w.shape[0]),min(32,w.shape[1]))
	gridDim = ((((w.shape[0] + blockDim[0]) - 1) / blockDim[0]), (((w.shape[1] + blockDim[1]) - 1) / blockDim[1]))

	d_update_weights[gridDim,blockDim](w,gw,lr)

#MOST SIMILAR

@cuda.jit('float32(float32,float32)',device=True)
def d_dist(a,b):
    return math.pow((b-a),2)

@cuda.jit('void(float32[:,:],float32[:,:],float32[:,:])')
def d_distances(a,b,val):
    x = cuda.grid(1)
    if (x < b.shape[0]):
		dist = 0.
		for y in range(b.shape[1]):
			dist += d_dist(a[0,y],b[x,y])
		dist = math.sqrt(dist)
		val[0,x] = dist


def most_similar(a):
	assert a.shape[1] == d_vectors.shape[1], "Size Mismatch: (%i,%i), (%i,%i)" %(a.shape[0],a.shape[1],d_vectors.shape[0],d_vectors.shape[1])
	blockDim = (1024)
	gridDim = (((d_vectors.shape[0] + blockDim) - 1) / blockDim)

	val = cuda.device_array((1,d_vectors.shape[0]),dtype='float32')

	d_distances[gridDim,blockDim](a,d_vectors,val)

	_,idx = margmin(val)

	return inv_vocab[idx]

#ARGMIN

@cuda.jit('void(float32[:,:],float32[:,:],int32[:,:])')
def d_margmin(a,val,idx):
    x = cuda.grid(1)
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
    total = min(cuda.blockDim.x, a.shape[1] - (cuda.blockIdx.x*cuda.blockDim.x))
    if (x < a.shape[1]):
		sVal = cuda.shared.array((1024),dtype=float32)
		sIdx = cuda.shared.array((1024),dtype=float32)
		s = total/2
		if x+s < total:
			if a[0,x] < a[0,x+s]:
				sVal[tidx] = a[0,x]
				sIdx[tidx] = x
			else:
				sVal[tidx] = a[0,x+s]
				sIdx[tidx] = x+s
			cuda.syncthreads()
			if total%2 == 1:
				if tidx == total-1:
					if a[0,x] < a[0,x-tidx]:
						sVal[0] = a[0,x]
						sIdx[0] = x
		s = s/2
		cuda.syncthreads()
		while s > 0:
			if tidx+s < total:
				if sVal[tidx] > sVal[tidx+s]:
					sVal[tidx] = sVal[tidx+s]
					sIdx[tidx] = sIdx[tidx+s]
				cuda.syncthreads()
				if s%2 == 1 and s > 1:
					if tidx == s-1:
						if sVal[0] > sVal[tidx]:
							sVal[0] = sVal[tidx]
							sIdx[0] = sIdx[tidx]
			s = s/2
		cuda.syncthreads()
		if tidx == 0:
			val[0,bidx] = sVal[tidx]
			idx[0,bidx] = sIdx[tidx]	

def margmin(a):

	blas.amin(a[0,:])

	'''
	blockDim = min(1024,a.shape[1])
	gridDim = (((a.shape[1]) + blockDim) - 1) / blockDim
	val = cuda.device_array_like(a)
	idx = cuda.device_array(a.shape,dtype='int32')
	d_margmin[gridDim,blockDim](a,val,idx)
	while gridDim > 1:
		last_gridDim = gridDim
		blockDim = gridDim
		gridDim = ((last_gridDim + blockDim) - 1) / blockDim
		d_msum[gridDim,blockDim](val,val,idx)
	return val[0,0],idx[0,0]
	'''

#ARGMAX

@cuda.jit('void(float32[:,:],float32[:,:],int32[:,:])')
def d_margmax(a,val,idx):
    x = cuda.grid(1)
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x
    total = min(cuda.blockDim.x, a.shape[1] - (cuda.blockIdx.x*cuda.blockDim.x))
    if (x < a.shape[1]):
		sVal = cuda.shared.array((1024),dtype=float32)
		sIdx = cuda.shared.array((1024),dtype=float32)
		s = total/2
		if x+s < total:
			if a[0,x] > a[0,x+s]:
				sVal[tidx] = a[0,x]
				sIdx[tidx] = x
			else:
				sVal[tidx] = a[0,x+s]
				sIdx[tidx] = x+s
			cuda.syncthreads()
			if total%2 == 1:
				if tidx == total-1:
					if a[0,x] > a[0,x-tidx]:
						sVal[0] = a[0,x]
						sIdx[0] = x
		s = s/2
		cuda.syncthreads()
		while s > 0:
			if tidx+s < total:
				if sVal[tidx] < sVal[tidx+s]:
					sVal[tidx] = sVal[tidx+s]
					sIdx[tidx] = sIdx[tidx+s]
				cuda.syncthreads()
				if s%2 == 1 and s > 1:
					if tidx == s-1:
						if sVal[0] < sVal[tidx]:
							sVal[0] = sVal[tidx]
							sIdx[0] = sIdx[tidx]
			s = s/2
		cuda.syncthreads()
		if tidx == 0:
			val[0,bidx] = sVal[tidx]
			idx[0,bidx] = sIdx[tidx]	

def margmax(a):

	return blas.amax(a[0,:])
'''
	blockDim = min(1024,a.shape[1])
	gridDim = (((a.shape[1]) + blockDim) - 1) / blockDim
	val = cuda.device_array_like(a)
	idx = cuda.device_array(a.shape,dtype='int32')
	d_margmax[gridDim,blockDim](a,val,idx)
	while gridDim > 1:
		last_gridDim = gridDim
		blockDim = gridDim
		gridDim = ((last_gridDim + blockDim) - 1) / blockDim
		d_msum[gridDim,blockDim](val,val,idx)
	return val[0,0],idx[0,0]'''

			
