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

def MSE_R2(target, prediction):
    '''
    Returns MSE and R2
    '''
    return MSE(target,prediction),R2(target,prediction)

def accuracy(target, prediction):
    '''
    Returns the accuracy score of prediction compared
    to target.
    '''
    #prediction = prediction.ravel()
    #target = target.ravel()
    #print(sum(prediction==target))
    #print(target.shape)
    return sum(prediction==target)/len(target)

METRIC_FUNC = {'r2':R2,
          'mse':MSE,
          'accuracy':accuracy}

def scores(target,prediction):
    '''
    Returns the MSE and R2 score for target and prediction
    '''
    return MSE(target,prediction),R2(target,prediction)

