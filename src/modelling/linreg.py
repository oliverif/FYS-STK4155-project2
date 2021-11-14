
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
        Fits the linreg model. Note that
        if fit_intercept is true, the
        function centers the data before fitting.
        This is important when using
        regularization as it avoids punishing
        the intercept.
        
        The offset is added back after the fit.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
            
        Output:
        -------
        self: LinReg object
            Outputs self for compatability with sklearn methods. 
            Also allows chained operations.
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
        '''  
        Returns prediction based on X.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        Outpus:
        -------
        pred: ndarray of shape (n_samples,1)
        '''
        return X @ self.beta + self.intercept 
    
    def score(self,X,z):
        '''
        Calculates the R2 score of the
        model.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
            
        Output:
        -------
        float: R2 score
        '''
        p = self.predict(X)
        return R2(z,p)      
        
    def _fit_ols(self,X,z): 
        '''
        Fit linear regression model without any regularization, AKA
        OLS regression.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
        '''
        self.beta = linalg.pinv(X.T @ X) @ X.T @ z
     
    def _fit_ridge(self,X,z):
        '''
        Fit linear regression model with l2 regularizer, AKA
        Ridge regression.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
        '''
        I =  eye(X.shape[1],X.shape[1]) #Identity matrix
        self.beta = linalg.pinv(X.T @ X + self.lmb*I) @ X.T @ z
        
    
        
class SGD_linreg(SGD_optimizer):
    '''
    Linear regressor with SGD optimizer.
    
    This model minimizes the squared error
    loss function to obtain optimal beta values
    and intercept.
    
    The class inherits from SGD_optimizer as it is the
    base class of all models.
    
    Parameters:
    ----------
    loss_func: str{'cross_entropy','squared_loss'} default='squared_loss'
        The loss function to use. Cross entropy is often used in classification
        problems while squared loss is used in regression.
        
    fit_intercept: bool, default=False
        Wether to fit the intercept or not. If false, it assumes the data
        is already centered.
        
    regularization: str{'l2',None} default='l2'
        Wether or not to use l2 regularization
        
    lmb: float, default = 0.001
        The regularization parameter. Only used if regularization is 'l2'
        
    momentum: float, default = 0.5
        The portion of the former velocity to influde in new velocity calculation.
        
    schedule: str{'constant','invscaling','decaying'}, default='constant'
        The learning rate schedule.
        
    lr0: float, default=0.01
        The inital learning rate. Only used by constant and invscaling.
        
    batch_size: int, default=32
        The amount of data points in each batch during mini batch stochastic
        gradient descent. Should be a multiplum of 2.
        
    n_epochs: int, default=100
        Number of epochs to train.
        
    t0: int, default=50
        parameter in the decaying schedule
    
    t1: int, default=300
        parameter in the decaying schedule
        
    power_t: float, default=0.05
        parameter in the invscaling schedule
        
    val_fraction: float, default=0.1
        The portion of the input data to set aside as validation data.
        Validation data is used to study loss and score during training.
    '''

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
                 t0 = 20,t1 = 100, 
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
        
        Inputs:
        -------
        shape: tuple
            Shape of design matrix
        '''
        #Reset beta
        self.beta = zeros(shape[1]).reshape(-1,1)
        #Reset intercept
        self.intercept = 0
        #Reset velocity
        self.v = 0


    def score(self,X,z):
        '''
        Calculates the R2 score of the
        model.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
            
        Output:
        -------
        float: R2 score
        '''
        p = self.predict(X)
        return R2(z,p)

    def predict(self,X):
        '''  
        Returns prediction based on X.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        Outpus:
        -------
        pred: ndarray of shape (n_samples,1)
        '''
        return X @ self.beta + self.intercept
      
    def predict_continuous(self,X):
        '''
        Outputs the prediction with continuous
        values. In this case(linear regression)
        it's exactly the same as predict. It's
        defined for generability in loss functions
        with other models.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        Output:
        -------
        pred: ndarray of shape (n_samples,1) 
        '''
        return self.predict(X)
      
    def partial_fit(self,X,z):
        '''
        Performs a single SGD step for linear model
        and updates beta and intercept accordingly.
        
        This function first predicts based on X to
        calculate an initial update value. 
        
        The mean of this value is the intercept update
        as it would be multiplied by 1 later anyways.
        Additionally, regularizing it is avoided.
        
        This update value is then inputted to the gradient
        function.
        
        If momentum is included, a velocity self.v is calculated
        and stored for this partial fit. Note that the velocity 
        from the former partial fit is used to update this
        velocity. The beta values are then added the velocity
        
        If momentum is not included, the beta values are
        subtracted the gradient directly.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
            
        Output:
        -------
        loss: float
            The loss captured right before beta and intercept
            update.
        
        '''
        p = self.predict(X)
        update = p - z
        #Update intercept
        if (self.fit_intercept):
            self.intercept -= self.lr*mean(update)

        #Add momentum
        if (self.momentum):    
            self.v = self.momentum*self.v - self.lr*self.cost_grad(X,update)
            self.beta += self.v          
        else:
            self.beta -= self.lr*self.cost_grad(X,update)
            
        return self.loss_func(z,p)

    def cost_grad(self, X, update):
        '''
        Gradient of squared loss cost function.
        
        This function calculates the analytical gradient
        of the squared error cost function.
        
        Regularization term is included depending
        on the specification upon initialization.
        
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
        grad = (2/X.shape[0])*(X.T @ update)
        if (self.regularization=='l2'):
            grad += (2/X.shape[0])*self.lmb*self.beta

        return grad
    
