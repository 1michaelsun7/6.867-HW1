import loadFittingDataP1 as loadfd
import loadParametersP1 as loadpm
import grad_descent as gd
import numpy as np
import math

def gradient_quadratic_bowl(A, B):
	def grad(x):
		return np.dot(A, x) - B
	return grad

def gradient_gaussian(f, mean, cov):
	def grad(x):
		return -np.dot(np.dot(f(x), np.linalg.inv(cov)),(x-mean))
	return grad

if __name__ == "__main__":
	#1.1
	(gaussMean,gaussCov,quadBowlA,quadBowlb) = loadpm.getData()
	def func(x):
		return 0.5*np.dot(np.transpose(x), np.dot(quadBowlA, x)) - np.dot(np.transpose(x), quadBowlb)
	grad_qb = gradient_quadratic_bowl(quadBowlA, quadBowlb)
	best_val = gd.functional_gradient_descent(func, np.random.random((2,)), f_prime=grad_qb)
	print best_val

	def gaussfunc(x):
		return -1.0/(math.sqrt(math.pow(2*math.pi, x.shape[0])*np.linalg.det(gaussCov)))*math.exp(-0.5*np.dot(np.dot(np.transpose(x-gaussMean),np.linalg.inv(gaussCov)),(x-gaussMean)))
	grad_gauss = gradient_gaussian(gaussfunc, gaussMean, gaussCov)
	best_val = gd.functional_gradient_descent(gaussfunc, np.ones((2,)))
	print best_val

	#1.2
	# By not initializing f_prime, we will use the central difference approximation in grad_descent.py
	(gaussMean,gaussCov,quadBowlA,quadBowlb) = loadpm.getData()
	def func(x):
		return 0.5*np.dot(np.transpose(x), np.dot(quadBowlA, x)) - np.dot(np.transpose(x), quadBowlb)
	grad_qb = gradient_quadratic_bowl(quadBowlA, quadBowlb)
	best_val = gd.functional_gradient_descent(func, np.random.random((2,)), f_prime=grad_qb)
	print best_val

	#1.3
	X, Y = loadfd.getData()
	array_x = np.array(X)
	array_y = np.array(Y)
	theta = gd.batch_gradient_descent(array_x, array_y, x_init=np.random.random((10,)), lr=1e-4, max_iters=10000)
	print theta

	print np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(array_x), array_x)), np.transpose(array_x)), array_y)

	#1.4
	X, Y = loadfd.getData()
	array_x = np.array(X)
	array_y = np.array(Y)
	theta = gd.stochastic_gradient_descent(array_x, array_y, x_init=np.random.random((10,)), lr=1e-2, lr2=10, max_iters=10000)
	print theta