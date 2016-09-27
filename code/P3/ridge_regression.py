import numpy as np
import math 
import random 

def ridge_regression(X, Y, M, lam):
	phi = basis(X, M)

	phi_square = np.dot(np.transpose(phi), phi)
	id_matrix = np.identity(phi_square.shape[0])

	phi_target = np.dot(np.transpose(phi), Y)

	weight_vect = np.dot(np.linalg.inv(lam*id_matrix + phi_square), phi_target)

	return phi, weight_vect

def basis(X, M):
	X = np.squeeze(X)
	M += 1
	N = len(X)
	phi = np.zeros((N,M))
	for col in xrange(M):
		phi[:,col] = np.power(X, col)
	return phi