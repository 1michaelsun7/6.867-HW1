import numpy as np
import math 
import random 
import matplotlib.pyplot as plt
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
	(X, Y) = loadfd.getData(False)
	#Set the different values of alpha to be tested
	alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

	best_loss = float('inf')
	best_params = (None, None, None)
	for alpha in alpha_ridge:
		phi, w = rr.ridge_regression(X, Y, 1, alpha)

		reg_error = 0.5 * np.sum(np.square(np.dot(phi, w) - Y)) + alpha*np.sum(np.square(w))
		if reg_error < best_loss:
			best_loss = reg_error
			best_params = (phi, w, alpha)

	print "Best Alpha: ", best_params[2]
	plot(np.dot(best_params[0], best_params[1]))
