import numpy as np
from scipy.special import xlogy, expit
from ..model_evaluation.metrics import MSE

def sigmoid(z):
    '''
    Sigmoid function.
    Uses scipy.special.expit to
    ensure stable output values.
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return expit(z)

def identity(z):
    '''
    This function does nothing
    to the input.
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return z
 
def relu(z):
    '''
    Rectified Linear Unit function.
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return np.maximum(z,0)

def leakyrelu(z):
    '''
    Leaky Rectified Linear Unit function.
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return np.where(z>=0,z,0.01*z)
       
def softmax(z):
    '''
    Softmax function.
    Only used for multiclass classification.
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return np.exp(z)/np.sum(np.exp(z),axis=0)

#Activation function dictionary
ACTIVATION_FUNCS = {'sigmoid':sigmoid,'identity':identity,'relu':relu,'leakyrelu':leakyrelu,'softmax':softmax}

def sigmoid_derivative(z,error):
    '''
    Derivative of sigmoid function
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    a = sigmoid(z)
    return a*(1-a)

def identity_derivative(z,error):
    '''
    Derivative of sigmoid function
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return z
 
def relu_derivative(z,error):
    '''
    Derivative of Rectified Linear Unit function
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return np.where(z<0,0,1)

def leakyrelu_derivative(z,error):
    '''
    Derivative of Leaky Rectified Linear Unit function
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
    '''
    return np.where(z>0,1,0.01)
       
def softmax_derivative(z,error):
    '''
    Derivative of Softmax function
    
    Input:
    -------
    z: ndarray of shape(n_samples,1)
        The signal to be activated.
        
    error: ndarray of shape(n_samples,1)
        The error from the former layer
    '''
    a = softmax(z)
    return a*(error-a)

#Activation function derivative dictionary
ACTIVATION_FUNCS_DERIVATIVE = {'sigmoid':sigmoid_derivative,
                               'identity':identity_derivative,
                               'relu':relu_derivative,
                               'leakyrelu':leakyrelu_derivative,
                               'softmax':softmax_derivative}

def squared_loss(z,p):
    '''
    Calculates the squared loss often used in regression.
    Loss function used is MSE divided by two.
    
    Inputs:
    -------
    z: ndarray of shape(n_samples,1)
        The target data
        
    p: ndarray of shape(n_samples,1)
        The predicted values
    '''
    return MSE(z,p)/2

def cross_entropy(z,p):
    '''
    Calculates the cross entropy loss, AKA 
    the logistic loss function.
    
    Inputs:
    -------
    z: ndarray of shape(n_samples,1)
        The target data
        
    p: ndarray of shape(n_samples,1)
        The predicted values
    '''
    return -np.sum(xlogy(z,p) + xlogy((1-z),(1-p)))/z.shape[0]

#Loss function dictionary
LOSS_FUNCS = {'squared_loss':squared_loss,
              'cross_entropy':cross_entropy}