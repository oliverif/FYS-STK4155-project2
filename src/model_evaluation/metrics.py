from numpy import sum,mean,size

def R2(target, prediction):
    '''
    Returns the R2 score of prediction compared
    to target.
    '''
    return 1 - sum((target - prediction) ** 2) / sum((target - mean(target)) ** 2)

def MSE(target, prediction):
    '''
    Returns the MSE of prediction compared 
    to target.
    '''
    n = size(prediction)
    return sum((target-prediction)**2)/n

def accuracy(target, prediction):
    '''
    Returns the accuracy score of prediction compared
    to target.
    '''
    return sum(prediction==target)/len(target)

def scores(target,prediction):
    '''
    Returns the MSE and R2 score for target and prediction
    '''
    return MSE(target,prediction),R2(target,prediction)

