from numbapro import vectorize
import numpy as np
import cProfile as profile

#@vectorize(["float32(float32,float32)"],target='gpu')
def but(x,y):
    return np.dot(x,y)

def run():
    a = np.random.rand(200,500)
    b = np.random.rand(500,200)
    c = but(a,b)

run()
#profile.run('run()')