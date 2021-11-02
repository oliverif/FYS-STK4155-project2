from modelling.ols import fit_beta
from numpy import random,mean, sum,zeros
from sklearn.utils import shuffle
from processing.data_preprocessing import center_data
from model_evaluation.metrics import MSE, R2
from modelling.common import predict


class SGD_optimizer:
    t0 = 50
    t1 = 300
    
    power_t = 0.05
    
    mse = 0
    r2 = 0

    def __init__(self,
                 regularization = 'l2',
                 lmb = 0.001, 
                 fit_intercept = False, 
                 use_momentum = True, 
                 gamma = 0.5,
                 schedule = 'constant',
                 lr0 = 0.01,
                 batch_size=None,
                 n_epochs=None):
        
        self.regularization = regularization
        self.lmb = lmb

        self.use_momentum = use_momentum
        self.gamma = gamma

        self.lr0 = lr0
        
        self.schedule = schedule
        self.vt = 0

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.intercept = 0
        self.fit_intercept = fit_intercept


        self.learning_schedule = self.set_schedule()
        self.cost_grad = self.set_cost_func()

        self.param_setters = {'lmb':self.set_lmb,
                              'regularization':self.set_cost_func,
                              'fit_intercept':self.set_fit_intercept,
                              'batch_size':self.set_batch_size,
                              'n_epochs':self.set_n_epochs,
                              'lr':self.set_lr,
                              'use_momentum':self.set_partial_fit_func,
                              'gamma':self.set_gamma}
        


    def fit(self,X_train, z_train, batch_size = None, n_epochs = None):
        '''
        Performs mini-batch stochastic gradient 
        descent optimization and stores resulting
        parameters in self.beta. AKA train model using X_train and
        z_train.
        '''
        #Set batch size and epochs if given
        if (batch_size is not None):
            self.set_batch_size(batch_size)
        if(n_epochs is not None):
            self.set_n_epochs(n_epochs)

        if(self.fit_intercept):
            self.intercept = 1

        #Initalize beta to random values
        #self.beta = random.randn(X_train.shape[1],1)

        #Initialize beta to zeros
        self.beta = zeros(X_train.shape[1]).reshape(-1,1)

        #Reset momentum
        self.vt = 0

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

    def set_gamma(self,gamma):
        self.gamma = gamma

    def set_lr(self, lr):
        '''
        Sets the learning rate
        '''
        self.lr = lr

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



    def set_fit_intercept(self,fit_intercept):
        self.fit_intercept = fit_intercept

    def score(self,X,z):
        '''
        Calculates the scores for this model
        '''
        z_tilde = predict(X,self.beta)
        self.mse = MSE(z,z_tilde)
        self.r2 = R2(z,z_tilde)

        return self.mse

    def get_params(self, deep=True):
        return {'lmb':self.lmb,
        'regularization':self.regularization,
        'fit_intercept':self.fit_intercept,
        'batch_size':self.batch_size,
        'n_epochs':self.n_epochs,
        'lr':self.lr,
        'gamma':self.gamma}

    def predict(self,X):
        return X @ self.beta + self.intercept

    def set_cost_func(self, regularization = None):
        '''
        Defines the cost function to be used during optimization
        '''
        if (regularization is not None):
            self.regularization = regularization

        if (self.regularization == 'l2'):
            return self.cost_grad_l2
        else:
            return self.cost_grad_l0
      
    def partial_fit(self,X,z):
        '''
        Single SGD step to updates parameters.
        '''
        update = self.predict(X) - z

        if (self.fit_intercept):
            self.intercept -= self.lr*mean(update)

        if (self.momentum):    
            self.vt = self.gamma*self.vt - self.lr*self.cost_grad(X,update)
            self.beta += self.vt
            
        else:
            self.beta -= self.lr*self.cost_grad(X,update)

    def cost_grad_l0(self, X, update):
        '''
        Gradient of squared loss cost function.
        '''
        return (1/X.shape[0])*(X.T @ update)

    def cost_grad_l2(self, X, update):
        '''
        Gradient of squared loss cost function with l2 regularizer.
        '''
        return (2/X.shape[0])*(X.T @ update) + (2/X.shape[0])*self.lmb*self.beta

    def fit_score(self,X_train, z_train, X_test, z_test):

        self.fit(X_train,z_train)
        self.score(X_train,z_train)
        z_pred = predict(X_test,self.beta)

        return self.mse, MSE(z_test,z_pred), self.r2, R2(z_test,z_pred)

