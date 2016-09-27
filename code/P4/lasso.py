import numpy as np
import math 
import random 

def lasso(x, y, lambda):
	phi = basis(x, 13)

def basis(x, M):
	phi = np.zeros((M,))

	phi[0] = x
	for i in xrange(1, M):
		phi[i] = math.sin(0.4*math.pi*i*x)