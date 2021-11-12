from numpy import random, arange
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from abc import abstractmethod

from src.visualization.visualize import plot_curves
from ._functions import LOSS_FUNCS



class SGD_optimizer(object):
    

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
        self.learning_schedule = self.set_schedule()
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


    def fit(self,X, z, batch_size = None, n_epochs = None):
        '''
        Performs mini-batch stochastic gradient 
        descent optimization.
        '''
        #Set batch size and epochs if given
        if (batch_size is not None):
            self.set_batch_size(batch_size)
        if(n_epochs is not None):
            self.set_n_epochs(n_epochs)        
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

        #The number of batches is calculated from batch size.
        n_batches = int(z.shape[0]/self.batch_size)
    
        for epoch in range(self.n_epochs):
            running_loss = 0          
            for batch in range(n_batches):
                #Select a random batch
                xi,zi = self.select_batch(X,z)
                #Update lr according to schedule
                self.learning_schedule(epoch*n_batches+batch)
                #Update parameters and capture loss
                running_loss += self.partial_fit(xi,zi)*(xi.shape[0])
                            
            #Calculate score and loss after epoch and store it
            self.loss.append(running_loss/X.shape[0])
            self.scores.append(self.score(X,z))
            if(self.val_fraction>0):
                p = self.predict_continuous(X_val)
                self.val_loss.append(self.loss_func(z_val,p))
                self.val_scores.append(self.score(X_val,z_val))
            #Shuffle training data for next round
            X,z = shuffle(X,z)

        return self

    #what
    def set_params(self,**new_params):
        '''
        Sets one or more parameters
        contained in dictionarey params
        '''
        for key, val in new_params.items():
            setattr(self,key,val)
        return self
 

    def get_params(self, deep=True):
        '''Gets parameters'''
        return {param:getattr(self,param) for param in self.params}

 
    def set_n_epochs(self,n_epochs):
        '''
        Sets number of epochs to be used during training
        '''
        self.n_epochs = n_epochs

    def set_batch_size(self,batch_size):
        '''
        Sets batch size to be used during training
        '''
        self.batch_size = batch_size

    def set_momentum(self,momentum):
        self.momentum = momentum

    def set_lr0(self, lr0):
        self.lr0 = lr0

    def set_regularization(self, regularization):
        self.regularization = regularization

    def set_lmb(self,lmb):
        '''
        Sets lambda used for regularization
        '''
        self.lmb = lmb

    def set_schedule(self,schedule=None):
        '''
        Returns a learning schedule based on input
        string.
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

    def set_decay_constants(self,t0,t1):
        '''
        Sets the constants to be used in learning schedule
        '''
        self.t0 = t0
        self.t1 = t1

    def decaying_schedule(self,t):
        '''
        Decaing schedule for learning rate.
        '''
        self.lr = self.t0/(t+self.t1)

    def invscaling_schedule(self,t):
        self.lr = self.lr0/pow(t+1,self.power_t)

    def const_schedule(self,t):
        '''
        Constant schedule(i.e no schedule) for learning rate.
        '''
        pass

    def select_batch(self, X, z):
        '''
        Selects a random batch from X and z with
        batch_size amount of data points.
        '''
        randi = random.randint(z.shape[0]/self.batch_size) #choose a random batch
        xi = X[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        zi = z[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        return xi, zi

    def plot_loss(self,title=None,ax=None):
        params = arange(1,self.n_epochs+1)
        if(self.val_fraction>0):
            ax = plot_curves({'Train loss':self.loss,
                            'Validation loss':self.val_loss},
                            params,
                            ('epoch','loss'),
                            title=title,
                            ax=ax)
        else:
            ax = plot_curves({'Train loss':self.loss},
                            params,
                            ('epoch','loss'),
                            title=title,
                            ax=ax)
        return ax
    
    def plot_score(self,title=None,ax=None):
        params = arange(1,self.n_epochs+1)
        if(self.val_fraction>0):
            ax = plot_curves({'Train score':self.scores,
                            'Validation score':self.val_scores},
                            params,
                            ('epoch','score'),
                            title=title,
                            ax=ax)
        else:
            ax = plot_curves({'Train score':self.scores},
                            params,
                            ('epoch','score'),
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
        
