from numpy import linalg, random,c_,zeros,insert
from model_evaluation.metrics import MSE
from processing.data_preprocessing import center_data
from autograd import grad
from sklearn.utils import shuffle
from modelling.common import *


def fit_beta(X_train,z_train,fit_intercept=True):
    '''
    Calculates optimal beta for OLS model from training data. 
    Returns beta(optimal parameters).
    '''

    #Center data
    if fit_intercept:
        X_train, X_offset = center_data(X_train)
        z_train,z_offset = center_data(z_train)

    
    beta = linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

    if fit_intercept:
        beta[0] = z_offset - X_offset @ beta #Intercept

    return beta


def predict(X_test, beta):
    '''
    Predicts target data based on test data.
    Intercept is added manually at the end.
    '''   
    return X_test[:,1:] @ beta[1:] + beta[0]

def fit_predict(X_train, z_train, X_test):
    '''
    Fits beta and predicts train test data in one go.
    '''
    beta = fit_beta(X_train, z_train)
    return predict(X_train, beta), predict(X_test,beta)

def cost_func( X, z_data, beta,lmb=None):
    '''
    Cost function is essentially the same as MSE
    however replaces input 'z_model' 
    with two inputs: X and beta.
    This enables autograd to derivate with respect
    to beta.
    '''
    return MSE(z_data,predict(X,beta))

def analytical_cost_grad(X,z,beta,lmb=None):
    '''
    Computes the analytical gradient
    at current X, z and beta values.
    '''
    return 2*(X.T @ (X @ beta-z))

def auto_cost_grad():
    '''
    Creates the gradient of cost function using
    autograd.
    Output: Gradient function
    '''
    return grad(cost_func,2)


def fit_beta_sgd(X_train, z_train, batch_size, n_epochs, lr='decaying' , gradient='auto'):
    '''
    Fit beta using stochastic gradient descent.
    X_train is shuffled between every epoch.
    '''
    beta = sgd(X_train, z_train, auto_cost_grad(), batch_size, n_epochs, lr)
    return beta

