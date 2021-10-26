from numpy import random,mean
from sklearn.utils import shuffle
from processing.data_preprocessing import center_data
from model_evaluation.metrics import MSE
from modelling.common import predict
from autograd import grad

class SGD_optimizer:
    t0 = 2
    t1 = 20
    vt = 0

    def __init__(self,regularization = 'l2',lmb = 0.001, fit_intercept = False, use_momentum = True, gamma = 0.5, lr = 'decaying',batch_size=None,n_epochs=None):
        self.regularization = regularization
        self.lmb = lmb

        self.use_momentum = use_momentum
        self.gamma = gamma

        self.lr = lr

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.intercept = 0
        self.fit_intercept = fit_intercept


        self.learning_schedule = self.set_schedule()
        self.cost_grad = self.set_cost_func()
        self.partial_fit = self.set_partial_fit_func()
        


    def fit(self,X_train, z_train, batch_size = None, n_epochs = None):
        '''
        Performs mini-batch stochastic gradient 
        descent optimization and returns resulting
        parameters. AKA train model using X_train and
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
        self.beta = random.randn(X_train.shape[1],1)

        n_batches = int(z_train.shape[0]/self.batch_size)

        for epoch in range(self.n_epochs):

            #Shuffle training data for next round
            X_train,z_train = shuffle(X_train,z_train)

            for batch in range(n_batches):

                #Select a random batch
                xi,zi = self.select_batch(X_train,z_train)

                #Update lr according to schedule
                self.learning_schedule(epoch*n_batches+batch)

                #Update beta
                self.partial_fit(xi,zi)

                
                
        return self.beta

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

    def set_partial_fit_func(self):
        '''
        Sets the partial fit function.
        The partial fit function is used
        to update the sgd optimization with
        one step. I.e subtract the gradient
        from the parameters once. 
        '''
        if(self.use_momentum):
            return self.sgd_step_momentum
        else:
            return self.sgd_step

            

    def set_lr(self, lr):
        '''
        Sets the learning rate
        '''
        self.lr = lr

    def set_schedule(self,lr=None):
        '''
        Returns a learning schedule based on input
        string.
        '''
        #If learning rate is given
        if(lr is not None):
            self.set_lr(lr)

        
        if(self.lr=='decaying'):
            return self.decaying_schedule
        elif(isinstance(self.lr,float) or isinstance(self.lr,int)):
            return self.const_schedule
        else:
            raise ValueError

    def set_decay_constants(self,t0,t1):
        '''
        Sets the constants to be used in learning schedule
        '''
        self.t0 = t0
        self.t1 = t1

    def momentum(self, X,z):
        update = self.calc_deviation(X,z)
        if (self.fit_intercept):
            self.fit_intercept += self.lr*update

        return self.gamma*self.vt + self.lr*self.cost_grad(X,update)

    def decaying_schedule(self,t):
        '''
        Decaing schedule for learning rate.
        '''
        self.lr = self.t0/(t+self.t1)


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

    def calc_deviation(self, X, z):
        '''
        Calculates the deviation bewteen prediction
        and target data.
        '''
        pred = X @ self.beta + self.intercept
        return pred-z


    def sgd_step(self,X,z):
        '''
        Updates beta using cost function gradiant and learning rate.
        '''
        update = self.calc_deviation(X,z)

        if (self.fit_intercept):
            self.intercept -= self.lr*mean(update)

        self.beta -= self.lr*self.cost_grad(X,update)

    def sgd_step_momentum(self,X,z):
        '''
        Updates beta with momentum using cost function gradiant and learning rate.
        '''
        update = self.calc_deviation(X,z)
        if (self.fit_intercept):
            self.intercept -= self.lr*mean(update)      

        self.vt = self.gamma*self.vt + self.lr*self.cost_grad(X,update)
        self.beta -= self.vt


    def cost_grad_l0(self, X, update):
        '''
        Gradient of squared loss cost function.
        '''
        return (2/X.shape[0])*(X.T @ update)

    def cost_grad_l2(self, X, update):
        '''
        Gradient of squared loss cost function with l2 regularizer.
        '''
        return (2/X.shape[0])*(X.T @ update) + 2*self.lmb*self.beta
