import numpy as np
from scipy.special import xlogy
from ..model_evaluation.metrics import MSE

def sigmoid(z):
    return(1/(1+np.exp(-z)))

def identity(z):
    return z
 
def relu(z):
    return np.maximum(z,0)

def leakyrelu(z):
    return np.where(z>=0,z,0.01*z)
       
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=0)

ACTIVATION_FUNCS = {'sigmoid':sigmoid,'identity':identity,'relu':relu,'leakyrelu':leakyrelu,'softmax':softmax}

def sigmoid_derivative(z,error):
    a = sigmoid(z)
    return a*(1-a)

def identity_derivative(z,error):
    return z
 
def relu_derivative(z,error):
    return np.where(z<0,0,1)

def leakyrelu_derivative(z,error):
    return np.where(z>0,1,0.01)
       
def softmax_derivative(z,error):
    a = softmax(z)
    return a*(error-a)

ACTIVATION_FUNCS_DERIVATIVE = {'sigmoid':sigmoid_derivative,
                               'identity':identity_derivative,
                               'relu':relu_derivative,
                               'leakyrelu':leakyrelu_derivative,
                               'softmax':softmax_derivative}

def squared_loss(z,p):
    '''
    Calculates the squared loss often used in regression.
    Loss function used is MSE divided by two.
    '''
    return MSE(z,p)/2

def cross_entropy(z,p):
    '''
    Calculates the cross entropy loss, AKA 
    the logistic loss function.
    '''
    return -np.sum(xlogy(z,p) + xlogy((1-z),(1-p)))/z.shape[0]

LOSS_FUNCS = {'squared_loss':squared_loss,
              'cross_entropy':cross_entropy}