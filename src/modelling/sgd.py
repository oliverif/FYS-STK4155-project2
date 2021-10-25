from numpy import random
from sklearn.utils import shuffle
from processing.data_preprocessing import center_data
from model_evaluation.metrics import MSE
from modelling.common import predict
from autograd import grad

class SGD_optimizer:
    t0 = 2
    t1 = 20
    vt = 0

    def __init__(self,regularization = 'l2',lmb = 0.001, use_momentum = True, gamma = 0.5, lr = 'decaying',batch_size=None,n_epochs=None):
        self.regularization = regularization
        self.lmb = lmb

        self.use_momentum = True
        self.gamma = gamma

        self.lr = lr

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.learning_schedule = self.set_schedule()
        self.cost_func = self.set_cost_func()
        self.cost_grad = self.set_grad()
        self.partial_fit = self.set_partial_fit_func()
        


    def fit(self,X_train, z_train, batch_size = None, n_epochs = None):
        '''
        Perform mini-batch stochastic gradient 
        descent optimization and returns resulting
        parameters. AKA train model using X_train and
        z_train.
        '''
        #Set batch size and epochs if given
        if (batch_size is not None):
            self.set_batch_size(batch_size)
        if(n_epochs is not None):
            self.set_n_epochs(n_epochs)

        #Center data before optimization for faster convergence.
        X_train, X_offset = center_data(X_train)
        z_train,z_offset = center_data(z_train)
        
        
        #Initalize beta to random values
        beta = random.randn(X_train.shape[1],1)

        #Set intercept to 0
        beta[0] = 0

        n_batches = int(z_train.shape[0]/self.batch_size)

        for epoch in range(self.n_epochs):
            X_train,z_train = shuffle(X_train,z_train)
            for batch in range(n_batches):

                #Select a random batch
                xi,zi = self.select_batch(X_train,z_train)

                #Update lr according to schedule
                self.lr = self.learning_schedule(epoch*n_batches+batch)

                #Update beta
                beta = self.partial_fit(xi,zi,beta)

                #Shuffle training data for next round
                

        #Ensure intercept has not exploded during optimization
        beta[0] = 0
        beta[0] = z_offset - X_offset @ beta #Intercept
        return beta

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

    def set_grad(self):
        '''
        Creates the gradient of cost function using
        autograd.
        Output: Gradient function
        '''
        return grad(self.cost_func,2)

    def set_cost_func(self, regularization = None):
        '''
        Defines the cost function to be used during optimization
        '''
        if (regularization is not None):
            self.regularization = regularization

        if (self.regularization == 'l2'):
            return self.cost_func_l2
        else:
            print("Cost")
            return self.cost_func_l0

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

    def l2_regularizer(self, beta):
        '''
        Calculates the regularizer term used in for instance Ridge
        '''
        return self.lmb*beta.T@beta

    def momentum(self, X,z,beta):

        return self.gamma*self.vt + self.lr*self.cost_grad(X,z,beta)

    def decaying_schedule(self,t):
        '''
        Decaing schedule for learning rate.
        '''
        return self.t0/(t+self.t1)


    def const_schedule(self,t):
        '''
        Constant schedule(i.e no schedule) for learning rate.
        '''
        return self.lr

    def select_batch(self, X_train, z_train):
        '''
        Selects a random batch from X_train and z_train with
        batch_size amount of data points.
        '''
        randi = random.randint(z_train.shape[0]/self.batch_size) #choose a random batch
        xi = X_train[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        zi = z_train[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        return xi, zi

    def sgd_step(self,X,z, beta):
        '''
        Updates beta using cost function gradiant and learning rate
        '''
        return beta - self.lr*self.cost_grad(X,z,beta)

    def sgd_step_momentum(self,X,z,beta):
        '''
        Updates beta using cost function gradiant and learning rate
        '''
        self.vt = self.momentum(X,z,beta)
        return beta - self.vt


    def cost_func_l0(self, X, z_data, beta):
        '''
        Cost function is essentially the same as MSE
        however replaces input 'z_model' 
        with two inputs: X and beta.
        This enables autograd to derivate with respect
        to beta.
        '''
        return MSE(z_data,predict(X,beta))

    def cost_func_l2(self, X, z_data, beta):
        '''
        Cost function for ridge MSE + l2.
        Replaces input 'z_model' 
        with two inputs: X and beta.
        This enables autograd to derivate with respect
        to beta.
        '''
        return MSE(z_data,predict(X,beta)) + self.l2_regularizer(beta)
