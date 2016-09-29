import numpy as np
import math 
import random 
import matplotlib.pyplot as plt
import lasso
import lassoData as LD

def plot(func):
    plt.plot(X,Y,'o')
    plt.plot(X,np.cos(np.pi*X)+np.cos(np.pi*2*X))
    plt.plot(X, func)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
	trainX, trainY = LD.lassoTrainData()
	valX, valY = LD.lassoValData()
	testX, testY = LD.lassoTestData()

	val_N = len(valX)
	val_phi = lasso.basis(valX, 13)

	best_val_loss = float('inf')
	best_lam = None
	best_w = None

	lambdas = [0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 0.7, 1]
	for cur_lam in lambdas:
		coeffs = np.array(lasso.lasso(trainX, trainY, cur_lam))
		loss = 0

		for i in xrange(val_N):
			loss += (valY[i] - np.dot(np.transpose(coeffs), val_phi[i]))**2

		loss /= val_N
		loss += cur_lam*np.sum(np.abs(coeffs))

		if loss < best_val_loss:
			print "Updating validation loss to ", round(loss, 4)
			best_val_loss = loss
			best_w = coeffs
			best_lam = cur_lam

	print "Best lambda: ", best_lam
	print "Best W: ", best_w

	test_N = len(testX)
	test_phi = lasso.basis(testX, 13)
	test_loss = 0
	for i in xrange(test_N):
		test_loss += (testY[i] - np.dot(np.transpose(best_w), test_phi[i]))**2

	test_loss /= test_N

	test_loss += best_lam*np.sum(np.abs(best_w))
	print "Test loss: ", test_loss

	true_w = LD.getTrueW()
	for i in xrange(test_N):
		test_loss += (testY[i] - np.dot(np.transpose(true_w), test_phi[i]))**2

	test_loss /= test_N

	test_loss += best_lam*np.sum(np.abs(true_w))
	print "True test loss: ", test_loss

	