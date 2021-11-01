from numpy import exp, random,full,matmul,zeros
from sklearn.utils import shuffle
import numpy as np


def sigmoid(z):
    return(1/(1+exp(-z)))

def linear(z):
    return z
    

ACTIVATION_FUNCS = {'sigmoid':sigmoid,'linear':linear}

class Layer:
    def __init__(self, weights, bias, activation):
        
        #activations from previous layer, i.e current state of this Layer
        #self.activations = zeros()
        
        #Weights of this layer
        self.weights = weights
        
        #Biases of this layer
        self.bias = bias

        self.activate = ACTIVATION_FUNCS[activation]
        
        


class NeuralNetwork:
    bias_init = 0.01
    def __init__(
            self,
            hidden_layer_sizes = (50,),
            hidden_activation = 'sigmoid',
            output_activation = 'linear',
            n_categories=1,
            n_epochs=10,
            batch_size=100,
            schedule = 'constant',
            lr0 = 0.01,
            use_momentum = True, 
            gamma = 0.5,
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
        return random.randn(n_features, n_neurons)
    
    def init_b(self,lenght):
        return full(lenght,self.bias_init)
    
    def feed_forward(self,X):
        '''
        Feeds X forward in the network and 
        stores each activations in the layers
        '''
        a = X
        for layer in self.layers:

            #Weighted sum
            z_h = matmul(a,layer.weights) + layer.bias
            
            #Store activations in layer object
            layer.activations = layer.activate(z_h)
            
            #Set activation as input for next iteration
            a = layer.activations
    
    def fast_feed_forwards(self,X):    
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

        return self.layers[-1].activations
    
    def backpropagation(self,X,z):
        
        #Feed forward to update layer activations
        self.feed_forward(X)
        
        #Do last layer first
        error = self.layers[-1].activations - z
        #Store weights for next layer
        weights = self.layers[-1].weights
        self._backpropagate(self.layers[-1],error)
        
        for layer in reversed(self.layers[:-1]):
            #Next layer error
            error = matmul(error, weights.T)*layer.activations*(1-layer.activations)
            #Store weights for next iteration
            weights = layer.weights
            #Back propagate one step
            self._backpropagate(layer,error)

    def _backpropagate(self,layer,error):
        #Calculate weights and biases gradients
        w_grad = self.calc_weight_grad(layer.activations, error)
        #print(w_grad.shape)
        b_grad = np.sum(error,axis=0)
        
        #Add l2 regularization if specified
        if(self.regularization=='l2'):

            w_grad += self.lmb * self.layers[-1].weights*self.layers[-1].weights.T
        #Update weights and biases
        layer.weights -= self.lr*w_grad
        layer.bias -= self.lr*b_grad

    def calc_weight_grad(self,activations,error):
        print(error.shape)
        return matmul(activations,error.T)
   
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
        
        return self.fast_feed_forwards(X)
     


