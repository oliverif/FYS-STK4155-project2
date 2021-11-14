from numpy import where
from ..model_evaluation.metrics import accuracy
from ._functions import sigmoid
from .linreg import SGD_linreg
from ._sgdBase import SGD_optimizer


class SGD_logreg(SGD_linreg,SGD_optimizer):

    def __init__(self,
                 loss_func = 'cross_entropy',
                 fit_intercept = False,
                 regularization = 'l2',
                 lmb = 0.001,
                 momentum = 0.5,
                 schedule = 'constant',
                 lr0 = 0.01,
                 batch_size=32,
                 n_epochs=100,
                 t0 = 50,t1 = 300, 
                 power_t = 0.05,
                 val_fraction = 0.1
                ):
    
        super().__init__(loss_func = loss_func,
                         fit_intercept=fit_intercept,
                         regularization = regularization,lmb = lmb,
                         momentum = momentum,
                         schedule = schedule,
                         lr0 = lr0,
                         batch_size = batch_size,
                         n_epochs = n_epochs,
                         t0 = t0, t1 = t1, 
                         power_t = power_t,
                         val_fraction=val_fraction
                        )

      
        #Additional parameters for this SGD linear regression
        self.v = 0
        self.fit_intercept = fit_intercept

    def predict(self,X):
        '''  
        Returns prediction based on X.
        
        This function classifies the predictions
        as well. It first uses LinReg's
        predict to predict continuous values,
        then runs them through the sigmoid function.
        
        Finally it calssifies the probabilities, based on
        p>0.5 = 1, and p<0.5 = 0.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        Outpus:
        -------
        pred: ndarray of shape (n_samples,1)
        '''
        #Use predict from linear regression first
        pred = super().predict(X)
        #and pass it through sigmoid
        pred = sigmoid(pred)
        return where(pred<0.5,0,1)
    
    def predict_continuous(self,X):
        '''
        Outputs the prediction with continuous
        values. AKA predicts probabilities.
        
        This function is similar as predict,
        however does not classify at the end.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        Outpus:
        -------
        pred: ndarray of shape (n_samples,1)
        '''
        #Use predict from linear regression first
        pred = super().predict(X)
        #and pass it through sigmoid
        return sigmoid(pred)
    
    def cost_grad(self,X,update):
        '''
        Gradient of log loss cost funtion.
        It's almost equal to the gradient of squared
        loss cost however without the factor 2.
        
        This function simply calls LinReg's
        cost_grad and divides it by two.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
            
        update: ndarray of shape(n_samples,1)
            Update value calculated in partial fit
        
        Output:
        -------
        grad: ndarray of shape(n_samples,1)
            Gradient of the beta values.
        '''

        return super().cost_grad(X,update) / 2
    
    def score(self,X,z):
        '''
        Returns the mean accuracy of the prediction.
        
        This function predicts based on X and compares its prediction with
        z.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data  
        '''
        #Predict
        p = self.predict(X)
        #Ensure correct shape
        if(len(z.shape)==1):
            z = z.reshape(-1,1)
        return accuracy(z,p)
    
    
    
