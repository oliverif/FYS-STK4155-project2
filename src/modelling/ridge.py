from numpy import linalg,eye
from processing.data_preprocessing import center_data
from model_evaluation.metrics import MSE

def fit_beta(X_train,z_train, lmb, fit_intercept=True):
    '''
    Calculates optimal beta for Ridge model from training data. 
    Returns beta(optimal parameters).

    Data is first centered to avoid penalization of intercept.
    Intercept is calculated separately and inserted back to beta array
    '''
    #Center data
    if fit_intercept:
        X_train, X_offset = center_data(X_train)
        z_train,z_offset = center_data(z_train)
    I =  eye(X_train.shape[1],X_train.shape[1]) #Identity matrix
    beta = linalg.pinv(X_train.T @ X_train + lmb*I) @ X_train.T @ z_train
    
    if fit_intercept:
        beta[0] = z_offset - X_offset @ beta #Intercept

    return beta

def predict(X_test, beta):
    '''
    Predicts target data based on test data.
    Intercept is added manually at the end.
    '''   
    return X_test[:,1:] @ beta[1:] + beta[0]

def fit_predict(X_train, z_train, X_test,lmb):
    '''
    Fits beta and predict test data in one go.
    '''   
    return predict(X_test,fit_beta(X_train, z_train,lmb))

def cost_func(X, z_data, beta):
    '''
    Cost function is essentially the same as MSE
    however replaces input 'z_model' 
    with two inputs: X and beta.
    This enables autograd to derivate with respect
    to beta.
    '''
    return MSE(z_data,predict(X,beta))


