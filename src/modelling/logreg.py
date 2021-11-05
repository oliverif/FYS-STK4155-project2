from numpy import where
from ..model_evaluation.metrics import accuracy
from .nn import sigmoid
from .linreg import SGD_linreg


class SGD_logreg(SGD_linreg):

    def __init__(self,
                 fit_intercept = False,
                 regularization = 'l2',
                 lmb = 0.001,
                 momentum = 0.5,
                 schedule = 'constant',
                 lr0 = 0.01,
                 batch_size=32,
                 n_epochs=10,
                 t0 = 50,t1 = 300, 
                 power_t = 0.05
                ):
    
        super().__init__(fit_intercept=fit_intercept,
                         regularization = regularization,lmb = lmb,
                         momentum = momentum,
                         schedule = schedule,
                         lr0 = lr0,
                         batch_size = batch_size,
                         n_epochs = n_epochs,
                         t0 = t0, t1 = t1, 
                         power_t = power_t,
                        )

      
        #Additional parameters for this SGD linear regression
        self.v = 0
        self.fit_intercept = fit_intercept
        self.param_setters['fit_intercept']= self.set_fit_intercept,

    def predict(self,X):
        '''
        Predicts output based on X
        '''
        pred = super().predict(X)
        pred = sigmoid(pred)
        return where(pred<0.5,0,1)
    
    def cost_grad(self,X,update):
        '''
        Gradient of log loss cost funtion.
        It's almost equal to the gradient of squared
        loss cost however without the factor 2.
        '''
        return super().cost_grad(X,update) / 2
    
    def score(self,X,z):
        p = self.predict(X)
        return accuracy(z,p)
    
    
    
