from numpy import random
from processing.data_preprocessing import center_data
from sklearn.utils import shuffle
from model_evaluation.metrics import scores


def predict(X_test, beta):
    '''
    Predicts target data based on test data.
    Intercept is added manually at the end.
    '''   
    return X_test[:,1:] @ beta[1:] + beta[0]

def predict_train_test(X_train,X_test,beta):
    return predict(X_train,beta),predict(X_test,beta)


def score_model(X_train,X_test,z_train,z_test,beta):
    t,p = predict_train_test(X_train,X_test,beta)
    return t,p, scores(z_train,t),scores(z_test,p)
