
from numpy import mean, zeros, linalg, eye

from ..processing.data_preprocessing import center_data
from ..model_evaluation.metrics import MSE, R2

from ._sgdBase import SGD_optimizer


class LinReg:
    def __init__(self,fit_intercept = False,
                 regularization = 'l2',
                 lmb = 0.001):
        self.regularization = regularization
        self.lmb=lmb
        self.fit_intercept = fit_intercept
        self.intercept = 0
        self.mse = 0
        self.r2 = 0
             
    def fit(self,X, z):
        '''
        Fits the linreg model
        '''
        if self.fit_intercept:
            #Center data
            X, X_offset = center_data(X)
            z,z_offset = center_data(z)
        
        if(self.regularization == 'l2'):
            #Ridge regression
            self._fit_ridge(X,z)
        else:
            #OLS regression
            self._fit_ols(X,z)
        
        if self.fit_intercept:
            #Calculating the intercept
            self.intercept = z_offset - X_offset @ self.beta     
        return self
     
    def predict(self,X):
        return X @ self.beta + self.intercept 
    
    def score(self,X,z):
        '''
        Calculates the R2 score of the
        model.
        '''
        p = self.predict(X,z)
        return R2(z,p)      
        
    def _fit_ols(self,X,z): 
        '''
        Fit linear regression model without any regularization, AKA
        OLS regression.
        '''
        self.beta = linalg.pinv(X.T @ X) @ X.T @ z
     
    def _fit_ridge(self,X,z):
        '''
        Fit linear regression model with l2 regularizer, AKA
        Ridge regression.
        '''
        I =  eye(X.shape[1],X.shape[1]) #Identity matrix
        self.beta = linalg.pinv(X.T @ X + self.lmb*I) @ X.T @ z
        
    
        
class SGD_linreg(SGD_optimizer):
      

    def __init__(self,
                 fit_intercept = False,
                 loss_func = 'squared_loss',
                 regularization = 'l2',
                 lmb = 0.001,
                 momentum = 0.5,
                 schedule = 'constant',
                 lr0 = 0.01,
                 batch_size=32,
                 n_epochs=100,
                 t0 = 2,t1 = 10, 
                 power_t = 0.05,
                 val_fraction = 0.1
                 ):
        
        super().__init__(loss_func=loss_func,
                         regularization = regularization,
                         lmb = lmb,
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
        self.params += ['fit_intercept']

        
    def initialize(self, shape):
        '''
        Initializes intercept and beta.
        Beta values and intercept are initialized to zero
        '''
        self.beta = zeros(shape[1]).reshape(-1,1)
        self.intercept = 0

    def set_fit_intercept(self,fit_intercept):
        self.fit_intercept = fit_intercept

    def score(self,X,z):
        '''
        Calculates the R score for this model
        '''
        p = self.predict(X)
        return R2(z,p)

    def get_params(self, deep=True):
        return {'lmb':self.lmb,
        'regularization':self.regularization,
        'fit_intercept':self.fit_intercept,
        'batch_size':self.batch_size,
        'n_epochs':self.n_epochs,
        'lr':self.lr,
        'lr0':self.lr0,
        'momentum':self.momentum}

    def predict(self,X):
        '''
        Predicts output based on X
        '''
        return X @ self.beta + self.intercept
      
    def predict_continuous(self,X):
        '''
        Outputs the prediction with continuous
        values. In this case(linear regression)
        it's exactly the same as predict. It's
        defined for generability in loss functions
        with other models.
        '''
        return self.predict(X)
      
    def partial_fit(self,X,z):
        '''
        Performs a single SGD step for linear model
        and updates beta and intercept accordingly.
        '''
        p = self.predict(X)
        update = p - z
        if (self.fit_intercept):
            self.intercept -= self.lr*mean(update)

        if (self.momentum):    
            self.v = self.momentum*self.v - self.lr*self.cost_grad(X,update)
            self.beta += self.v          
        else:
            self.beta -= self.lr*self.cost_grad(X,update)
            
        return self.loss_func(z,p)

    def cost_grad(self, X, update):
        '''
        Gradient of squared loss cost function.
        '''
        grad = (2/X.shape[0])*(X.T @ update)
        if (self.regularization=='l2'):
            grad += (2/X.shape[0])*self.lmb*self.beta

        return grad
    
