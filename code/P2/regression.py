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

def approx_gradient(f, init, delta):
#    print (f(init + delta) - f(init - delta))
    return (f(init + delta) - f(init - delta))/(2*delta)

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
    