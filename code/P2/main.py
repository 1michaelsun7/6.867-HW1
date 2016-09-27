import loadFittingDataP2 as loadfd
import numpy as np
import matplotlib.pyplot as plt


"""
Created on Mon Sep 26 23:04:30 2016

@author: Laura Pang
"""
def f(X, Y, M):  
    M+=1
    print 'X: ', X
    print 'Y: ', Y
    N = len(X)
    phi = np.zeros((N,M))
    for col in xrange(M):
        phi[:,col] = np.power(X, col)
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
        tot+= (Y[j] - np.dot(phi[j], w))*phi[j] ##?????
    return tot
    
def plot(func):
##    xran = numpy.linspace(0,1,num=400)
    plt.plot(X,Y,'o')
    plt.plot(X,np.cos(np.pi*X)+np.cos(np.pi*2*X))
    plt.plot(X, func)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    
if __name__ == "__main__":
    #2.1
    (X, Y) = loadfd.getData(False)
    phi = f(X, Y, 3)
    w = np.dot(np.linalg.inv(np.dot(np.transpose(phi), phi)),np.dot(np.transpose(phi), Y))
    print 'w: ', w
    plot(np.dot(phi,w))
    
    #2.2
    sse = SSE(X, Y, w, phi)
    print sse