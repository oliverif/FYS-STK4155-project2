from abc import abstractproperty
from modelling.ols import fit_beta
from numpy import random,mean, sum,zeros
from sklearn.utils import shuffle
from processing.data_preprocessing import center_data
from model_evaluation.metrics import MSE, R2
from modelling.common import predict
from abc import ABCMeta,abstractmethod




class SGD_optimizer(object):
    

    def __init__(self,
                 regularization = 'l2',
                 lmb = 0.001, 
                 momentum = 0.5,
                 schedule = 'constant',
                 lr0 = 0.01,
                 batch_size=None,
                 n_epochs=None,
                 t0 = 50,t1 = 300, 
                 power_t = 0.05,  
                 ):
        
        self.regularization = regularization
        self.lmb = lmb

        self.momentum = momentum

        self.lr0 = lr0       
        self.schedule = schedule

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.learning_schedule = self.set_schedule()

        self.t0 = t0
        self.t1 = t1
        self.power_t = power_t
        
        self.param_setters = {'lmb':self.set_lmb,
                              'regularization':self.set_regularization,
                              'batch_size':self.set_batch_size,
                              'n_epochs':self.set_n_epochs,
                              'lr':self.set_lr,
                              'lr0':self.set_lr0,
                              'momentum':self.set_momentum}
        
    @abstractmethod
    def initialize(self,shape):
        '''Initialize weights, biases and or other paremeters'''


    def fit(self,X_train, z_train, batch_size = None, n_epochs = None):
        '''
        Performs mini-batch stochastic gradient 
        descent optimization.
        '''
        #Set batch size and epochs if given
        if (batch_size is not None):
            self.set_batch_size(batch_size)
        if(n_epochs is not None):
            self.set_n_epochs(n_epochs)

        #Initialize parameters
        self.initialize(X_train.shape)

        #The number of batches is calculated from batch size.
        n_batches = int(z_train.shape[0]/self.batch_size)

        for epoch in range(self.n_epochs):
            for batch in range(n_batches):
                #Select a random batch
                xi,zi = self.select_batch(X_train,z_train)

                #Update lr according to schedule
                self.learning_schedule(epoch*n_batches+batch)

                #Update parameters
                self.partial_fit(xi,zi)

            #Shuffle training data for next round
            X_train,z_train = shuffle(X_train,z_train)

        return self

    def set_params(self,**params):

        for key, val in params.items():
            self.param_setters[key](val)

        return self
 
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

    def set_lr(self, lr):
        '''
        Sets the learning rate
        '''
        self.lr = lr

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

    def select_batch(self, X_train, z_train):
        '''
        Selects a random batch from X_train and z_train with
        batch_size amount of data points.
        '''
        randi = random.randint(z_train.shape[0]/self.batch_size) #choose a random batch
        xi = X_train[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        zi = z_train[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        return xi, zi

    @abstractmethod
    def score(self,X,z):
        '''Calculates the scores'''
        
    @abstractmethod
    def get_params(self, deep=True):
        '''Gets parameters'''

    @abstractmethod
    def predict(self,X):
        '''Predict output from X'''
      
    @abstractmethod
    def partial_fit(self,X,z):
        '''Single SGD step to updates parameters.'''
        

