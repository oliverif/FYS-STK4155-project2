from numpy import linalg, random,c_,zeros,insert
from model_evaluation.metrics import MSE
from processing.data_preprocessing import center_data
from autograd import grad
from sklearn.utils import shuffle



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

def cost_func( X, z_data, beta):
    '''
    Cost function is essentially the same as MSE
    however replaces input 'z_model' 
    with two inputs: X and beta.
    This enables autograd to derivate with respect
    to beta.
    '''
    return MSE(z_data,predict(X,beta))

def analytical_cost_grad(X,z,beta):
    '''
    Computes the analytical gradient
    at current X, z and beta values.
    '''
    n = z.shape[0]
    return 2/n*(X.T @ (X @ beta-z))

def auto_cost_grad():
    '''
    Creates the gradient of cost function using
    autograd.
    Output: Gradient function
    '''
    return grad(cost_func,2)


def update_beta_sgd(X,z, beta, cost_grad, lr=0.001):
    '''
    Updates beta using cost function gradiant and learning rate
    '''
    return beta - lr*cost_grad(X,z,beta)

def fit_beta_sgd(X_train, z_train, lr, batch_size, n_epochs, gradient='auto'):
    '''
    Fit beta using stochastic gradient descent.
    X_train is shuffled between every epoch.
    '''
    #X_train = X_train[:,1:]

    X_train, X_offset = center_data(X_train)
    z_train,z_offset = center_data(z_train)

    
    #initalize beta to random values
    beta = random.randn(X_train.shape[1],1)
    beta[0] = 0
    cost_gradient = auto_cost_grad()

    n = z_train.shape[0]
    n_batches = int(z_train.shape[0]/batch_size)

    for epoch in range(n_epochs):
        for batch in range(n_batches):
            randi = random.randint(n_batches) #choose a random batch
            xi = X_train[randi*batch_size : randi*batch_size+batch_size]
            zi = z_train[randi*batch_size : randi*batch_size+batch_size]
            beta = update_beta_sgd(xi,zi,beta,cost_gradient,lr)
            X_train,z_train = shuffle(X_train,z_train)

    #X_train = c_[zeros(X_train.shape[0]),X_train]
    beta[0] = z_offset - X_offset @ beta#Intercept
    return beta

