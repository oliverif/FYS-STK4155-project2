from sklearn.model_selection import KFold
from numpy import mean

def cross_validate(model, X, z, k_folds, X_scaler=None, z_scaler = None):
    '''
    Calculates the cross validated score of model.
    The scoring metric  is given by the model itself.
    Ex. 
    model is a classifier -> metric is accuracy.
    model is a regressor -> metric is R2.
    '''
    
    kfold = KFold(n_splits = k_folds, shuffle=True) 
    train_scores = []
    test_scores = []
    
    for train_inds, test_inds in kfold.split(X):
        X_train = X[train_inds]
        z_train = z[train_inds]

        X_test = X[test_inds]
        z_test = z[test_inds]
        
        if(X_scaler is not None):
            X_train = X_scaler.fit_transform(X_train)
            X_test = X_scaler.transform(X_test)
 
        if(z_scaler is not None):
            z_train = z_scaler.fit_transform(z_train)
            z_test = z_scaler.transform(z_test) 
        
        model.fit(X_train, z_train)

        train_scores.append(model.score(X_train, z_train))
        test_scores.append(model.score(X_test, z_test))
    
    return dict(train_scores=train_scores, test_scores=test_scores)

def cross_val_score(model, X, z, k_folds, X_scaler=None, z_scaler = None):
    '''
    Runs cross_validate and calculates average score across
    k folds
    '''
    scores = cross_validate(model, 
                            X, 
                            z, 
                            k_folds, 
                            X_scaler, 
                            z_scaler)
    return mean(scores['train_scores']), mean(scores['test_scores'])