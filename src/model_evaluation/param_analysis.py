from numpy import zeros
from sklearn.model_selection import GridSearchCV
import pandas as pd

def evaluate_parameter(X_train, z_train, X_test, z_test, param, vals, sgd):
    '''
    Fits an sgd optimizer using for every 
    'param' value in 'vals'.
    Returns MSE and/or R2 score arrays for the different
    parameter values.
    
    Inputs
    -----------
    param: {'lambda', 'lr', 'n_epochs','n_batches'}
        Parameter to evaluate

    vals: ndarray of shape (n_params,)
        Array containing different values of param to evaluate

    sgd_optimizer: SGD_optimizer
        SGD optimizer object used to fit weights with above parameter
    '''

    #Number of parameters to test
    num_vals = vals.shape[0]

    #Arrays to store scores
    mse_train = zeros((num_vals))
    mse_test = zeros((num_vals))
    r2_train = zeros((num_vals))
    r2_test = zeros((num_vals))

    #Dictionary to choose setter function
    param_func_dict = {'lambda':sgd.set_lmb,
    'lr':sgd.set_lr,
    'n_epochs':sgd.set_n_epochs,'n_batches':sgd.set_batch_size}

    set_param = param_func_dict[param]

    #Fit with every val and store MSE and R2
    for i in range(vals.shape[0]):
        set_param(vals[i])
        mse_train[i], mse_test[i], r2_train[i], r2_test[i] = sgd.fit_score(X_train,z_train, X_test, z_test)

    return mse_train, mse_test, r2_train, r2_test

    
def grid_search_df(X, z, model, param_grid):
    '''
    Performs a grid search for best model
    performance across param_grid. 
    Funcion wraps sklearn GridSearchCV and
    outputs more readable results.
    '''
    gs = GridSearchCV(estimator = model, 
                      param_grid = param_grid,
                      n_jobs=-1)
    gs = gs.fit(X,z)
    param_strs =['param_'+s for s in list(param_grid.keys())]
    data = {k[6:]: gs.cv_results_[k] for k in (param_strs)}
    data['mean_test_score'] = gs.cv_results_['mean_test_score']
    data['rank_test_score'] = gs.cv_results_['rank_test_score']

    return gs, pd.DataFrame(data)