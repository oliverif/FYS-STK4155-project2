from numpy import random
from sklearn.utils import shuffle
from modelling.ridge import cost_func
from processing.data_preprocessing import center_data
from model_evaluation.metrics import MSE
from modelling.common import predict
from autograd import grad

class SGD_optimizer:
    t0 = 2
    t1 = 20
    vt = 0

    def __init__(self,regularization = 'l2',lmb = 0.001, use_momentum = True, lr = 'decaying',batch_size=None,n_epochs=None, ):
        self.regularization = regularization
        self.use_momentum = True
        self.lr = lr
        self.lmb = lmb
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.cost_func = self.set_cost_func()




    
    def set_grad(self):
        

        return grad(cost_func,2)

    def set_cost_func(self, regularization = None):
        if (regularization is not None):
            self.regularization = regularization

        if (regularization == 'l2'):
            return self.cost_func_l2
        else:
            return self.cost_func


    def set_lr(self, lr):
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

    def momentum(X,z, beta, cost_grad,lmb, lr, vt,gamma):
        return gamma*vt + lr**cost_grad(X,z,beta,lmb)

    def decaying_schedule(self,t):
        '''
        Decaing schedule for learning rate.
        '''
        return self.t0/(t+self.t1)


    def const_schedule(lr,t):
        '''
        Constant schedule(i.e no schedule) for learning rate.
        '''
        return lr

    def select_batch(X_train, z_train, batch_size):
        '''
        Selects a random batch from X_train and z_train with
        batch_size amount of data points.
        '''
        randi = random.randint(z_train.shape[0]/batch_size) #choose a random batch
        xi = X_train[randi*batch_size : randi*batch_size+batch_size]
        zi = z_train[randi*batch_size : randi*batch_size+batch_size]
        return xi, zi

    def sgd_step(X,z, beta, cost_grad,lmb, lr=0.001, vt=None, gamma=None):
        '''
        Updates beta using cost function gradiant and learning rate
        '''
        return beta - lr*cost_grad(X,z,beta,lmb), vt

    def cost_func(self, X, z_data, beta):
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
