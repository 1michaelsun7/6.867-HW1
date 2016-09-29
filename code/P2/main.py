import loadFittingDataP2 as loadfd
import numpy as np
import matplotlib.pyplot as plt
import regression as rg

"""
Created on Mon Sep 26 23:04:30 2016

@author: Laura Pang
"""

def plot(X1, func):
##    xran = numpy.linspace(0,1,num=400)
    plt.plot(X,Y,'o')
    plt.plot(X1,np.cos(np.pi*X1)+np.cos(np.pi*2*X1))
    plt.plot(X1, func)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.rcParams.update({'font.size': 50})
    plt.show()
  
if __name__ == "__main__":
    #2.1
    # M=10
    (X, Y) = loadfd.getData(False)
    # phi = rg.f(X, Y, M)
    # w = np.dot(np.linalg.inv(np.dot(np.transpose(phi), phi)),np.dot(np.transpose(phi), Y))
    # print 'w: ', w
    
    # xran = np.linspace(0,1,num=400)
    # phi2 = rg.f(xran, Y, M)
#    print np.dot(phi2, w)
#    plot(xran, np.dot(phi2, w))
    
    #2.2
    # sse = rg.SSE(X, Y, w, phi)
    # ssed = rg.SSEDeriv(X, Y, w, phi)
    
    # def fn(w1):
    #     return rg.SSE(X,Y,w1,phi)

    print rg.approx_gradient(fn, w, 1e-6)
    
    #2.4
    phi = rg.fCos(X, Y)
    w = np.dot(np.linalg.inv(np.dot(np.transpose(phi), phi)),np.dot(np.transpose(phi), Y))
    print 'w: ', w
#    plot(np.dot(phi,w))
    
#    xran = np.linspace(0,1,num=400)
#    phi2 = rg.fCos(xran, Y)
#    print np.dot(phi2, w)
#    plot(xran, np.dot(phi2, w)) ###SMOOTH CUURVES
    
    # print rg.approx_gradient(fn, w, 1e-6)

    #2.3
    array_x = np.array(X)
    array_y = np.array(Y)
    for i in xrange(5):
        M = i+1
        phi = rg.f(X, Y, M)
        # theta, SSEs = rg.batch_gradient_descent(phi, array_y, array_x, x_init=np.random.random((M+1,)), lr=0.05, max_iters=1e8)
        # print theta
        # plt.plot(range(len(SSEs)), SSEs)
        # plt.xlabel('iters')
        # plt.ylabel('SSE')
        # plt.show()

        theta = rg.stochastic_gradient_descent(phi, array_y, array_x, x_init=np.zeros((M+1,)), lr=2e-2, max_iters=1e8)
        print theta
    
    #2.4
#     phi = rg.fCos(X, Y)
#     w = np.dot(np.linalg.inv(np.dot(np.transpose(phi), phi)),np.dot(np.transpose(phi), Y))
#     print 'w: ', w
# #    plot(np.dot(phi,w))
    
#     xran = np.linspace(0,1,num=400)
#     phi2 = rg.fCos(xran, Y)
#     print np.dot(phi2, w)
#     plot(xran, np.dot(phi2, w)) ###SMOOTH CUURVES