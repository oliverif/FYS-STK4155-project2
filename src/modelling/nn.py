from numpy import exp, random,full,matmul,zeros
from sklearn.utils import shuffle
import numpy as np


def sigmoid(z):
    return(1/(1+np.exp(-z)))

def identity(z):
    return z
 
def relu(z):
    return np.maximum(z,0)

def leakyrelu(z):
    return np.where(z>=0,z,0.01*z)
       
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=0)

ACTIVATION_FUNCS = {'sigmoid':sigmoid,'identity':identity,'relu':relu,'leakyrelu':leakyrelu,'softmax':softmax}

def sigmoid_derivative(z,error):
    a = sigmoid(z)
    return a*(1-a)

def identity_derivative(z,error):
    return z
 
def relu_derivative(z,error):
    return np.where(z<0,0,1)

def leakyrelu_derivative(z,error):
    return np.where(z>0,1,0.01)
       
def softmax_derivative(z,error):
    a = softmax(z)
    return a*(error-a)

ACTIVATION_FUNCS_DERIVATIVE = {'sigmoid':sigmoid_derivative,
                               'identity':identity_derivative,
                               'relu':relu_derivative,
                               'leakyrelu':leakyrelu_derivative,
                               'softmax':softmax_derivative}

class Layer:
    def __init__(self, weights, bias, activation):
        
        #activations from previous layer, i.e current state of this Layer
        #self.activations = zeros()
        
        #Weights of this layer
        self.weights = weights
        
        #Biases of this layer
        self.bias = bias
        
        #Velocities
        self.v_w = 0
        self.v_b = 0

        self.activate = ACTIVATION_FUNCS[activation]
        self.derivative = ACTIVATION_FUNCS_DERIVATIVE[activation]
        
        


class NeuralNetwork:
    bias_init = 0.01
    def __init__(
            self,
            hidden_layer_sizes = (50,),
            hidden_activation = 'sigmoid',
            output_activation = 'identity',
            n_categories=1,
            n_epochs=10,
            batch_size=32,
            schedule = 'constant',
            w_init='uniform',
            b_init = 0.001,
            lr0 = 0.01,
            momentum = 0.5,
            regularization = 'l2',
            lmb = 0.001, 
            ):


        self.hidden_layer_sizes = hidden_layer_sizes
        
        #+1 for outputlayer
        self.n_layers = len(hidden_layer_sizes)+1
        self.layers = [None]*(len(hidden_layer_sizes)+1)
        self.weights = [None]*(len(hidden_layer_sizes)+1)
        self.biases = [None]*(len(hidden_layer_sizes)+1)
        #self.n_inputs = X_data.shape[0]
        #self.n_features = X_data.shape[1]
        self.momentum = momentum
        self.lr = lr0
        self.w_init = w_init
        self.b_init = b_init
        self.n_categories = n_categories
        self.regularization = regularization

        self.learning_schedule = self.const_schedule
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        #self.iterations = self.n_inputs // self.batch_size
        self.lr0 = lr0
        self.lmb = lmb
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        

    def initalize_layers(self,input_shape):
        '''
        Initializes layers given input shape. Used
        when fit is called.
        '''
        self.n_samples,self.n_features = input_shape
        
        prev_shape = (self.hidden_layer_sizes[0],input_shape[1])

        for layer,neurons in enumerate(self.hidden_layer_sizes):
            w = self.init_w(prev_shape[1],neurons)
            b = self.init_b(neurons)
            self.layers[layer] = Layer(w,b,self.hidden_activation)

            prev_shape = w.shape
         
        #Adding output layer   
        w = self.init_w(prev_shape[1],self.n_categories)
        b = self.init_b(self.n_categories)
        self.layers[-1] = Layer(w,b,self.output_activation)
                 
    def init_w(self,n_features,n_neurons):
        if (self.w_init=='normal'):
            return random.randn(n_features, n_neurons)
        elif(self.w_init=='uniform'):
            return random.uniform(0,1,(n_features, n_neurons))
        elif(self.w_init =='glorot'):
            limit = np.sqrt(6.0/(n_features+n_neurons))
            return random.uniform(-limit,limit,(n_features, n_neurons))
    
    def init_b(self,lenght):
        return full(lenght,self.b_init)
    
    def feed_forward(self,X):
        '''
        Feeds X forward in the network and 
        stores each activations in the layers
        '''
        a = X

        for layer in self.layers:

            #Weighted sum
            layer.z_h = matmul(a,layer.weights) + layer.bias
            
            #Store activations in layer object
            layer.activations = layer.activate(layer.z_h)

            a = layer.activations
            
            #Set activation as input for next iteration
            #a = layer.activations
    
    def fast_feed_forward(self,X):    
        '''
        Feeds X forward in the network and
        returns output layer activation
        without storing intermitten activations
        '''             
        a = X
        for layer in self.layers:
            #Weighted sum
            z_h = matmul(a,layer.weights) + layer.bias
            
            #Store activations in layer object
            a = layer.activate(z_h)
            
            #Set activation as input for next iteration

        return a
    
    def backpropagation(self,X,z):
        
        #Feed forward to update layer activations
        self.feed_forward(X)
        
        #Do last layer first
        error = self.layers[-1].activations - z

        #Store weights for next layer
        weights = self.layers[-1].weights


        w_grad, b_grad = self._calc_grads(weights,self.layers[-2].activations, error)

        self._update_weights_and_biases(self.layers[-1],w_grad,b_grad)
        
        for i in range(len(self.layers)-2,0,-1):

            #Next layer error
            #error = matmul(error, weights.T)*self.layers[i].derivative(self.layers[i].z_h)
            error = (error @ weights.T)*self.layers[i].derivative(self.layers[i].z_h)
            #Store weights for next iteration
            weights = self.layers[i].weights

            w_grad, b_grad = self._calc_grads(weights,self.layers[i-1].activations, error)
            
            #Back propagate one step
            self._update_weights_and_biases(self.layers[i],w_grad,b_grad)
            
            
        #First layer
        error = matmul(error, weights.T)*self.layers[0].derivative(self.layers[0].z_h)
        w_grad, b_grad = self._calc_grads(self.layers[0].weights, X, error)
        self._update_weights_and_biases(self.layers[0],w_grad,b_grad)
     
     
    def _update_weights_and_biases(self,layer,w_grad,b_grad):
        '''
        Updates the weights and biases of layer.
        Either updates with momentum or simply
        subtracts w_grad and b_grad from
        weights and biases respectively.
        '''
        
        if (self.momentum):    
            layer.v_w = self.momentum*layer.v_w - self.lr*w_grad
            layer.v_b = self.momentum*layer.v_b - self.lr*b_grad
            layer.weights += layer.v_w
            layer.bias += layer.v_b
            
        else:
            layer.weights -=  self.lr*w_grad
            layer.bias -=  self.lr*b_grad
         
       
    def _calc_grads(self,current_layer_w,prev_layer_a, error):
        #Calculate weights and biases gradients
        w_grad = matmul(prev_layer_a.T,error)
        b_grad = np.mean(error,axis=0)


        #Add l2 regularization if specified
        if(self.regularization=='l2'):
            w_grad += self.lmb * current_layer_w
        
        w_grad /=self.batch_size
        
        return w_grad,b_grad


    def calc_weight_grad(self,activations,error):

        return matmul(activations.T,error)
   
   
   
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
    
    def fit(self,X_train,z_train,batch_size = None, n_epochs = None):
        '''
        Performs mini-batch stochastic gradient 
        descent optimization and stores resulting
        weights and biases in Layers. 
        AKA train model using X_train and z_train.
        '''
        #Set batch size and epochs if given
        if (batch_size is not None):
            self.set_batch_size(batch_size)
        if(n_epochs is not None):
            self.set_n_epochs(n_epochs)


        #Initialize layers
        self.initalize_layers(X_train.shape)

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

                #Update weights
                self.backpropagation(xi,zi)

            #Shuffle training data for next round
            X_train,z_train = shuffle(X_train,z_train)

        return self

    
    def select_batch(self, X_train, z_train):
        '''
        Selects a random batch from X_train and z_train with
        batch_size amount of data points.
        '''
        randi = random.randint(z_train.shape[0]/self.batch_size) #choose a random batch
        xi = X_train[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        zi = z_train[randi*self.batch_size : randi*self.batch_size+self.batch_size]
        return xi, zi
    
    def predict(self,X):
        
        return self.fast_feed_forward(X)
     
    def set_n_epochs(self,n_epochs):
        self.n_epochs = n_epochs

