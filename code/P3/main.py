import numpy as np
import math 
import random 
import matplotlib.pyplot as plt
import regressData as rgd
import loadFittingDataP2 as loadfd
import ridge_regression as rr

def plot(func):
    plt.plot(X,Y,'o')
    plt.plot(X,np.cos(np.pi*X)+np.cos(np.pi*2*X))
    plt.plot(X, func)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":
	# 3.1
	(X, Y) = loadfd.getData(False)
	# #Set the different values of alpha to be tested
	# alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

	# best_loss = float('inf')
	# best_params = (None, None, None)
	# for alpha in alpha_ridge:
	# 	phi, w = rr.ridge_regression(X, Y, 1, alpha)

	# 	reg_error = 0.5 * np.sum(np.square(np.dot(phi, w) - Y)) + alpha*np.sum(np.square(w))
	# 	if reg_error < best_loss:
	# 		best_loss = reg_error
	# 		best_params = (phi, w, alpha)

	# print "Best Alpha: ", best_params[2]
	# plot(np.dot(best_params[0], best_params[1]))

	# 3.2
	A_trainX, A_trainY = rgd.regressAData()
	B_trainX, B_trainY = rgd.regressBData()
	valX, valY = rgd.validateData()

	# params
	alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
	M_vals = range(10)

	best_loss = float('inf')
	best_M = None
	best_alpha = None
	best_w = None

	# # model select for A
	# for M in M_vals:
	# 	for alpha in alpha_ridge:
	# 		phi, w = rr.ridge_regression(A_trainX, A_trainY, M, alpha)
	# 		val_phi = rr.basis(valX, M)

	# 		val_error = 0.5 * np.sum(np.square(np.dot(val_phi, w) - valY)) + alpha*np.sum(np.square(w))

	# 		if val_error < best_loss:
	# 			print "Loss updated to ", round(val_error, 4)
	# 			best_loss = val_error
	# 			best_M = M
	# 			best_alpha = alpha
	# 			best_w = w

	# # test on B data
	# phi_B = rr.basis(B_trainX, best_M)
	# test_loss = 0.5 * np.sum(np.square(np.dot(phi_B, best_w) - B_trainY)) + best_alpha*np.sum(np.square(best_w))
	# print "Test set loss: ", test_loss
	# print "Best params: ", (M, alpha)

	# model select for B
	for M in M_vals:
		for alpha in alpha_ridge:
			phi, w = rr.ridge_regression(B_trainX, B_trainY, M, alpha)
			val_phi = rr.basis(valX, M)

			val_error = 0.5 * np.sum(np.square(np.dot(val_phi, w) - valY)) + alpha*np.sum(np.square(w))

			if val_error < best_loss:
				print "Loss updated to ", round(val_error, 4)
				best_loss = val_error
				best_M = M
				best_alpha = alpha
				best_w = w

	# test on A data
	phi_A = rr.basis(A_trainX, best_M)
	test_loss = 0.5 * np.sum(np.square(np.dot(phi_A, best_w) - A_trainY)) + best_alpha*np.sum(np.square(best_w))
	print "Test set loss: ", test_loss
	print "Best params: ", (M, alpha)

