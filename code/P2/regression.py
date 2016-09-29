import matplotlib.pyplot as plt
import numpy as np
import random

"""
Created on Wed Sep 28 01:16:31 2016

@author: Laura Pang
"""

def f(X, Y, M):  
    M+=1
#    print 'X: ', X
#    print 'Y: ', Y
    N = len(X)
    phi = np.zeros((N,M))
    for col in xrange(M):
        phi[:,col] = np.power(X, col)
#    print phi
    return phi
    
def fCos(X, Y):  
    M=8
#    print 'X: ', X
#    print 'Y: ', Y
    N = len(X)
    phi = np.zeros((N,M))
    for col in xrange(M):
        phi[:,col] = np.cos(X*np.pi*(col+1))
#    print phi
    return phi

def SSE(X, Y, w, phi):
    N = len(X)
    tot = 0
    for j in range(N):
        tot+= (Y[j] - np.dot(phi[j], w))**2
    return 0.5*tot

def SSEDeriv(X, Y, w, phi):
    N = len(X)
    tot = 0
    for j in range(N):
        tot+= (Y[j] - np.dot(phi[j], w))*-phi[j]
    return tot
    

### code from grad_descent in P1
def approx_gradient(f, init, delta):
    return (f(init + delta) - f(init - delta))/(2*delta)
    
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