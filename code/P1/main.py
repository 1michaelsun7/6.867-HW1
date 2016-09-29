import loadFittingDataP1 as loadfd
import loadParametersP1 as loadpm
import grad_descent as gd
from scipy.stats import multivariate_normal
import numpy as np
import math
import matplotlib.pyplot as plt

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
	# (gaussMean,gaussCov,quadBowlA,quadBowlb) = loadpm.getData()
	# def func(x):
	# 	return 0.5*np.dot(np.transpose(x), np.dot(quadBowlA, x)) - np.dot(np.transpose(x), quadBowlb)
	# grad_qb = gradient_quadratic_bowl(quadBowlA, quadBowlb)
	# best_val, norms = gd.functional_gradient_descent(func, np.random.random((2,)), f_prime=grad_qb, lr=1, max_diff=1e-16)
	# print best_val
	# print np.dot(quadBowlA, best_val) - quadBowlb

	# def gaussfunc(x):
	# 	return np.negative(multivariate_normal.pdf(x, gaussMean, gaussCov))
	# grad_gauss = gradient_gaussian(gaussfunc, gaussMean, gaussCov)
	# best_val, norms = gd.functional_gradient_descent(gaussfunc, np.random.random((2,)),  f_prime=grad_gauss, lr=1e6, max_diff=1e-8)
	# print best_val
	# plt.plot(range(len(norms)), norms)
	# plt.xlabel('iters')
	# plt.ylabel('norm')
	# plt.show()

	#1.2
	# By not initializing f_prime, we will use the central difference approximation in grad_descent.py
	# (gaussMean,gaussCov,quadBowlA,quadBowlb) = loadpm.getData()
	# def func(x):
	# 	return 0.5*np.dot(np.transpose(x), np.dot(quadBowlA, x)) - np.dot(np.transpose(x), quadBowlb)
	# grad_qb = gradient_quadratic_bowl(quadBowlA, quadBowlb)
	# best_val = gd.functional_gradient_descent(func, np.random.random((2,)), f_prime=grad_qb)
	# print best_val

	#1.3
	# X, Y = loadfd.getData()
	# array_x = np.array(X)
	# array_y = np.array(Y)
	# theta = gd.batch_gradient_descent(array_x, array_y, x_init=np.random.random((10,)), lr=1e-4, max_iters=10000)
	# print theta

	# print np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(array_x), array_x)), np.transpose(array_x)), array_y)

	# #1.4
	X, Y = loadfd.getData()
	array_x = np.array(X)
	array_y = np.array(Y)
	theta = gd.stochastic_gradient_descent(array_x, array_y, x_init=np.random.random((10,)), lr=1000000, max_iters=10000)
	print theta
