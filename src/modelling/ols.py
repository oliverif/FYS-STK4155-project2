from numpy import linalg
from model_evaluation.metrics import MSE
from autograd import grad

def fit_beta(X_train,z_train):
    '''
    Calculates optimal beta for OLS model from training data. 
    Returns beta(optimal parameters).
    '''
    return linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train


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

def cost_func( X, z_data, beta):
    '''
    Cost function is essentially the same as MSE
    however replaces input 'z_model' 
    with two inputs: X and beta.
    This enables autograd to derivate with respect
    to beta.
    '''
    return MSE(z_data,predict(X,beta))


def update_beta_sgd(beta, grad, lr=0.001):
    '''
    Updates beta using gradiant and learning rate
    '''
    return beta - lr*grad(beta)
