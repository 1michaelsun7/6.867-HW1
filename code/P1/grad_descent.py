import matplotlib.pyplot as plt
import numpy as np
import random

def functional_gradient_descent(f, init, f_prime=None, lr=0.01, max_iters=10000, max_diff=1e-6):
	iters = 0
	
	best_value = float("inf")

	while iters < max_iters:
		eval_fn = f(init)
		cur_value = np.linalg.norm(eval_fn)

		deriv = f_prime(init) if f_prime else approx_gradient(f, init, 1e-6)

		if abs(cur_value-best_value) < max_diff or np.linalg.norm(deriv) < max_diff:
			best_value = cur_value
			break
		
		init -= lr*deriv

		if cur_value < best_value:
			best_value = cur_value

		iters += 1

	print "Converged after %d iterations" % iters
	return init

def approx_gradient(f, init, delta):
	return (f(init + delta) - f(init - delta))


def batch_gradient_descent(x, y, x_init=[None], lr=0.01, max_iters=10000):
	iters = 0
	
	# epsilon for convergence criterion
	eps = 1e-6

	# information about the data
	num_samples = x.shape[0]

	# initialize theta (subject to change)
	init_zeros = np.zeros((x.shape[1],))
	theta = x_init if x_init.any() else init_zeros

	J_err = np.linalg.norm(np.dot(x,theta)-y)**2

	while iters < max_iters:
		if iters % 1000 == 0:
			print "Iteration %d" % iters

		# d/dTheta - since this is batch, we take the sum of the gradient across all data points
		grad_theta = np.zeros((x.shape[1],))
		for j in xrange(x.shape[1]):
			grad_theta[j] = np.sum(1.0/float(num_samples)*(np.dot(x,theta)-y)*x[:,j])

		theta -= lr*grad_theta
		new_J_err = np.linalg.norm(np.dot(x,theta)-y)**2

		if abs(new_J_err - J_err) < eps or np.linalg.norm(grad_theta) < eps:
			print "Converged after %d iterations with loss %f" % (iters, new_J_err)
			J_err = new_J_err
			break

		J_err = new_J_err
		if iters == max_iters - 1:
			print "Max iterations (%d iterations) exceeded" % max_iters
		iters += 1

	return theta

def stochastic_gradient_descent(x, y, x_init=[None], lr=0.01, max_iters=10000):
	iters = 0
	
	# epsilon for convergence criterion
	eps = 1e-6

	# information about the data
	num_samples = x.shape[0]

	# initialize theta (subject to change)
	init_zeros = np.zeros((x.shape[1],))
	theta = x_init if x_init.any() else init_zeros

	J_err = np.linalg.norm(np.dot(x,theta)-y)**2

	while iters < max_iters:
		if iters % 1000 == 0:
			print "Iteration %d" % iters

		# d/dTheta - since this is batch, we take the sum of the gradient across all data points
		grad_theta = np.zeros((x.shape[1],))
		for j in xrange(x.shape[1]):
			grad_theta[j] = np.sum(1.0/float(num_samples)*(np.dot(x,theta)-y)*x[:,j])

		theta -= lr*grad_theta
		new_J_err = np.linalg.norm(np.dot(x,theta)-y)**2

		if abs(new_J_err - J_err) < eps or np.linalg.norm(grad_theta) < eps:
			print "Converged after %d iterations with loss %f" % (iters, new_J_err)
			J_err = new_J_err
			break

		J_err = new_J_err
		if iters == max_iters - 1:
			print "Max iterations (%d iterations) exceeded" % max_iters
		iters += 1

	return theta
