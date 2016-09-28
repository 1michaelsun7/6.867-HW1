import numpy as np
import math 
import random 
from sklearn import linear_model as sklin

def lasso(x, y, lam):
	phi = basis(x, 13)

	clf = sklin.Lasso(alpha = lam)
	clf.fit(phi, y)

	return clf.coef_

def basis(x, M):
	x = np.squeeze(x)
	N = x.shape[0]
	phi = np.zeros((N,M))

	phi[:,0] = x
	for i in xrange(1, M):
		phi[:,i] = np.sin(0.4*np.pi*i*x)

	return phi