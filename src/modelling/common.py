from numpy import random
from processing.data_preprocessing import center_data
from sklearn.utils import shuffle


def predict(X_test, beta):
    '''
    Predicts target data based on test data.
    Intercept is added manually at the end.
    '''   
    return X_test[:,1:] @ beta[1:] + beta[0]

def decaying_schedule(lr,t,t0=2,t1=20):
    '''
    Decaing schedule for learning rate.
    '''
    return t0/(t+t1)

def const_schedule(lr,t):
    '''
    Constant schedule(i.e no schedule) for learning rate.
    '''
    return lr

def choose_schedule(schedule):
    '''
    Returns a learning schedule based on input
    string.
    '''
    if (schedule=='decaying'):
        return decaying_schedule
    elif(isinstance(schedule,float) or isinstance(schedule,int)):
        return const_schedule
    else:
        raise ValueError

def select_batch(X_train, z_train, batch_size):
    '''
    Selects a random batch from X_train and z_train with
    batch_size amount of data points.
    '''
    randi = random.randint(z_train.shape[0]/batch_size) #choose a random batch
    xi = X_train[randi*batch_size : randi*batch_size+batch_size]
    zi = z_train[randi*batch_size : randi*batch_size+batch_size]
    return xi, zi


def l2_regularizer(beta,lmb):
    '''
    Calculates the regularizer term used in for instance Ridge
    '''
    return lmb*beta.T@beta


def momentum(X,z, beta, cost_grad,lmb, lr, vt,gamma):
    return gamma*vt + lr*cost_grad(X,z,beta,lmb)


def sgd_step_momentum(X,z, beta, cost_grad,lmb, lr, vt, gamma):
    '''
    Updates beta using cost function gradiant and learning rate
    '''
    vt = momentum(X,z, beta, cost_grad,lmb, lr, vt, gamma)
    return beta - vt, vt

def sgd_step(X,z, beta, cost_grad,lmb, lr=0.001, vt=None, gamma=None):
    '''
    Updates beta using cost function gradiant and learning rate
    '''
    return beta - lr*cost_grad(X,z,beta,lmb), vt

def sgd(X_train, z_train, cost_gradient, batch_size, n_epochs, lr, lmb=None, momentum=False, gamma=0.5):
    '''
    Generic stochastic gradient descent.
    X_train is shuffled between every epoch.
    '''
    #X_train = X_train[:,1:]

    #X_train, X_offset = center_data(X_train)
    #z_train,z_offset = center_data(z_train)
    vt = 0
    
    #initalize beta to random values
    beta = random.randn(X_train.shape[1],1)
    #beta[0] = 0
    if(momentum):
        sgd_update = sgd_step_momentum
    else:
        sgd_update = sgd_step
        gamma = None

    learning_schedule = choose_schedule(lr)
    n_batches = int(z_train.shape[0]/batch_size)



    for epoch in range(n_epochs):
        X_train,z_train = shuffle(X_train,z_train)
        for batch in range(n_batches):
            xi,zi = select_batch(X_train,z_train,batch_size)
            lr = learning_schedule(lr,epoch*n_batches+batch)
            beta,vt = sgd_update(xi,zi,beta,cost_gradient,lmb,lr,vt,gamma)
            

    #beta[0] = 0
    #beta[0] = z_offset - X_offset @ beta #Intercept
    return beta
