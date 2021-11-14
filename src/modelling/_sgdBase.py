from numpy import random, arange
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from abc import abstractmethod

from src.visualization.visualize import plot_curves
from ._functions import LOSS_FUNCS



class SGD_optimizer(object):
    '''
    Base class for all SGD based models.
    
    This class defines all common methods
    for models using sgd optimization.
    
    Note that the class should not be used on
    it's own and is only meant as a blueprint
    for other models.
    
    Ensure all abstract methods are defined in other models.
    '''
    def __init__(self,
                 loss_func,
                 regularization,
                 lmb, 
                 momentum,
                 schedule,
                 lr0,
                 batch_size,
                 n_epochs,
                 t0,t1, 
                 power_t,
                 val_fraction  
                 ):
        self.loss_func = LOSS_FUNCS[loss_func]
        self.regularization = regularization
        self.lmb = lmb
        self.momentum = momentum
        self.lr0 = lr0
        self.lr = lr0       
        self.schedule = schedule
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.t0 = t0
        self.t1 = t1
        self.power_t = power_t
        self.val_fraction = val_fraction 
        self.scores = []
        self.loss = []
        self.val_scores = []
        self.val_loss = []
        self.params = ['regularization',
                       'lmb',
                       'momentum',
                       'schedule',
                       'lr0',
                       'batch_size',
                       'n_epochs',
                       't0', 
                       'power_t' 
                       ]
        
    @abstractmethod
    def initialize(self,shape):
        '''Initialize weights, biases and or other paremeters'''


    def fit(self,X, z):
        '''
        Performs mini-batch stochastic gradient 
        descent optimization.
        
        The main part of this function contains
        a double for loop. The inner loop
        iterates the batches, i.e a new batch for each
        iteration, while the outer loop iterates epochs.
        As such, n_batches is iterated across n_epochs
        amount of times. Note that n_samples of data points
        is always processed in each epoch.
        
        Within the inner for loop the batch is selected,
        the learning rate is set and a partial fit is
        conducted. 
        
        The partial fit also returns a loss
        for a particular batch. Note that
        the loss is multiplied by the batch size
        as the batch_size may change if n_samples
        is not divisible by batch_size. As such, a small
        batch size doesn't contribute as much as a larger
        to the loss.
        
        Loss and score are then captured.
        
        Note that this function shuffles the data set by default
        as to avoid fitting to any spurious correlations as a result
        of the order of the data.
        
        Validation can also be employed. Here the function
        sets aside a fraction of the given data to test on after each
        epoch.

        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
        
        Output:
        -------
        self: _sgdBase object
            Outputs self for compatability with sklearn methods. 
            Also allows chained operations.

        '''
      
        #Ensure correct shape  
        if(len(z.shape) == 1):
            z = z.reshape(-1,1)         
        #Split data into train and validation if val_fraction > 0
        if (self.val_fraction>0):
            X, X_val, z, z_val = train_test_split(X,z,test_size=self.val_fraction)

        #Reset the learning rate, score and loss in case of refit
        self.lr = self.lr0
        self.loss = []
        self.scores = []
        self.val_loss = []
        self.val_scores = []
        
        #Initialize parameters
        self.initialize(X.shape)
        #Set schedule
        self.learning_schedule = self.set_schedule()

        #The number of batches is calculated from batch size.
        n_batches = int(z.shape[0]/self.batch_size)
    
        for epoch in range(self.n_epochs):
            #Shuffle training data
            X,z = shuffle(X,z)
            running_loss = 0          
            for batch in range(n_batches):
                #Select a random batch
                xi,zi = self.select_batch(X,z)
                #Update lr according to schedule
                self.learning_schedule(epoch*n_batches+batch)
                #Update parameters and capture loss
                running_loss += self.partial_fit(xi,zi)*(xi.shape[0])
                            
            #Calculate score and loss after epoch and store it
            self.loss.append(running_loss/z.shape[0])
            self.scores.append(self.score(X,z))
            #Capture validation metrics
            if(self.val_fraction>0):
                p = self.predict_continuous(X_val)
                self.val_loss.append(self.loss_func(z_val,p))
                self.val_scores.append(self.score(X_val,z_val))
            
            

        return self


    def set_params(self,**new_params):
        '''
        Sets one or more parameters
        contained in dictionarey params.
        Uses pythons setattr to set attribute.
        
        Input:
        ------
        new_params: dict(paramname:value)
        
        Output:
        ------
        self: _sgdBase object
            Outputs self for compatability. Also
            allows chained functions.
        '''
        for key, val in new_params.items():
            setattr(self,key,val)
        return self
 
    def get_params(self, deep=True):
        '''
        Gets parameters using pythons getattr
        function. Only return those also included
        in the parameter list self.params.
        
        Input:
        ------
        deep(not in use): bool
            This is not in used, but added
            for compatability with sklearn methods.
        '''
        return {param:getattr(self,param) for param in self.params}

    def set_schedule(self,schedule=None):
        '''
        Returns a learning schedule based on input
        string. Used to set the learning schedule
        function for the model.
        
        Input:
        ------
        schedule(optional): str
            The schedule string that determines
            the schedule function.
        
        Output:
        ------
        schedule function: python function
            The actual schedule function
        '''
        
        if(schedule is not None):
            self.schedule = schedule

        #Also set the learning rate function
        if(self.schedule=='decaying'):
            return self.decaying_schedule
        elif(self.schedule=='constant'):
            self.lr = self.lr0
            return self.const_schedule
        elif(self.schedule =='invscaling'):
            return self.invscaling_schedule
        else:
            raise ValueError

    #Learning schedule functions
    def decaying_schedule(self,t):
        '''
        Decaying schedule for learning rate.
        
        Input:
        ------
        t: int
            Typically n_epochs*n_batches+batch in training loop
            
        '''
        self.lr = self.t0/(t+self.t1)

    def invscaling_schedule(self,t):
        '''
        Inverse scaling for learning rate.
        
        Input:
        ------
        t: int
            Typically n_epochs*n_batches+batch in training loop
            
        '''
        self.lr = self.lr0/pow(t+1,self.power_t)

    def const_schedule(self,t):
        '''
        Constant schedule(i.e no schedule) for learning rate.
        This function does nothing to change the learning rate.
        '''
        pass

    def select_batch(self, X, z):
        '''
        Selects a random batch from X and z with
        batch_size amount of data points.
        Note that if n_samples(z.shape[0]) is not
        divisible by batch_size, the last batch will
        contain n_samples%batch_size samples.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
            
        Output:
        -------
        xi: ndarray(batch_size,n_features)
        
        zi: ndarray(batch_size,1)
        '''
        #When n_samples is not divisible by batch size
        #add an extra possible value in the random
        #generator. Note that multi indexing to beyond the
        #array in numpy simply returns the rest of the array. 
        if(z.shape[0]%self.batch_size):   
            randi = self.batch_size* random.randint(int(z.shape[0]/self.batch_size)+1) #choose a random batch
        else:
            randi = self.batch_size* random.randint(int(z.shape[0]/self.batch_size)) #choose a random batch
        xi = X[randi : randi+self.batch_size]
        zi = z[randi : randi+self.batch_size]
        return xi, zi

    def plot_loss(self,title=None,ax=None):
        '''
        Plots the loss curve generated during training.
        This function employs plot_curves from 
        src.visualization.visualize
        
        Input:
        ------
        title(optional): str
            The title of the plot
            
        ax(optional): matplotlib.axes object
            The ax to draw the plot in
            
        Output:
        ------
        ax: matplotlib.axes object
            The ax containing the plot
        '''
        #Create epochs array as corresponding x-axis values
        epochs = arange(1,self.n_epochs+1)
        
        #Plot validation loss if included
        if(self.val_fraction>0):
            ax = plot_curves({'Train loss':self.loss,
                            'Validation loss':self.val_loss},
                            epochs,
                            ('Number of completed epochs','Loss'),
                            title=title,
                            ax=ax)
        else:
            ax = plot_curves({'Train loss':self.loss},
                            epochs,
                            ('Number of completed epochs','Loss'),
                            title=title,
                            ax=ax)
        return ax
    
    def plot_score(self,title=None,ax=None):
        params = arange(1,self.n_epochs+1)
        if(self.val_fraction>0):
            ax = plot_curves({'Train score':self.scores,
                            'Validation score':self.val_scores},
                            params,
                            ('Number of completed epochs','Score'),
                            title=title,
                            ax=ax)
        else:
            ax = plot_curves({'Train score':self.scores},
                            params,
                            ('Number of completed epochs','Score'),
                            title=title,
                            ax=ax)            
        return ax
        

    @abstractmethod
    def score(self,X,z):
        '''Calculates the scores'''
        
    @abstractmethod
    def predict(self,X):
        '''Predict output from X'''
      
    @abstractmethod
    def predict_continuous(self,X):
        '''Predict continous output from X'''
            
    @abstractmethod
    def partial_fit(self,X,z):
        '''Single SGD step to updates parameters.'''
        
