import pdb
import random
import pylab as pl

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def lassoTrainData():
    return getData('lasso_train.txt')

def lassoValData():
    return getData('lasso_validate.txt')

def lassoTestData():
    return getData('lasso_test.txt')

def getTrueW():
	data = pl.loadtxt('lasso_true_w.txt')
	return data