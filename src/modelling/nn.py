from numpy import exp, random,full,matmul,zeros
from ..model_evaluation.metrics import R2, accuracy
import numpy as np
from ._sgdBase import SGD_optimizer
from ._functions import ACTIVATION_FUNCS, ACTIVATION_FUNCS_DERIVATIVE

class Layer:
    def __init__(self, weights, bias, activation):
                
        #Weights of this layer
        self.weights = weights
        
        #Biases of this layer
        self.bias = bias
        
        #Velocities
        self.v_w = 0
        self.v_b = 0

        #activate function will create class variable
        #self.activations = activate(z_h)
        self.activate = ACTIVATION_FUNCS[activation]
        self.derivative = ACTIVATION_FUNCS_DERIVATIVE[activation]
        
        


class NeuralNetwork(SGD_optimizer):
    def __init__(
            self,
            hidden_layer_sizes = (50,),
            hidden_activation = 'relu',
            output_activation = 'sigmoid',
            n_categories=1,
            w_init='glorot',
            b_init = 0.001,
            loss_func = 'cross_entropy',
            regularization = 'l2',
            lmb = 0.001,
            momentum = 0.5,
            schedule = 'constant',
            lr0 = 0.01,
            batch_size=32,
            n_epochs=100,
            t0=50, t1=300, 
            power_t=0.05,
            val_fraction=0.1
            ):

        super().__init__(loss_func = loss_func,
                         regularization = regularization,
                         lmb = lmb,
                         momentum = momentum,
                         schedule = schedule,
                         lr0 = lr0,
                         batch_size = batch_size,
                         n_epochs = n_epochs,
                         t0 = t0, t1 = t1, 
                         power_t = power_t,
                         val_fraction=val_fraction               
                         )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_layers = len(hidden_layer_sizes)+1 #+1 for outputlayer
        self.layers = [None]*(len(hidden_layer_sizes)+1)
        self.weights = [None]*(len(hidden_layer_sizes)+1)
        self.biases = [None]*(len(hidden_layer_sizes)+1)
        self.w_init = w_init
        self.b_init = b_init
        self.n_categories = n_categories        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        #Store model parameters in list for easy access
        #Enables the use of SKlearn CVGridsearch
        new_params = ['hidden_layer_sizes',
                      'hidden_activation',
                      'output_activation']     
        self.params += new_params
        
    def initialize(self,input_shape):
        '''
        Initializes layers given input shape. Used
        when fit is called.
        '''
        self.lr = self.lr0
        self.n_samples,self.n_features = input_shape
        prev_shape = (self.hidden_layer_sizes[0],input_shape[1])
        for layer,neurons in enumerate(self.hidden_layer_sizes):
            w = self._init_w(prev_shape[1],neurons)
            b = self._init_b(neurons)
            self.layers[layer] = Layer(w,b,self.hidden_activation)
            prev_shape = w.shape
         
        #Adding output layer   
        w = self._init_w(prev_shape[1],self.n_categories)
        b = self._init_b(self.n_categories)
        self.layers[-1] = Layer(w,b,self.output_activation)
  
    def partial_fit(self,X,z):
        '''
        Performs a single SGD step for neural network
        and updates weights and biases accordingly.
        '''
        p = self._feed_forward(X)
        self._backpropagation(X,z)
        loss = self.loss_func(z,p)
        if(self.regularization == 'l2'):
            for layer in self.layers:
                w = layer.weights.ravel()
                loss += self.lmb * w @ w / (2*X.shape[0])             
        return loss
    
    def score(self,X,z):
        '''
        Returns the coefficient of determination(R2) for the prediction if
        NN is regressor.
        Returns the mean accuracy of the prediction if NN is classifier.
        '''
        p = self.predict(X)
        if(len(z.shape)==1):
            z = z.reshape(-1,1)        
        if (self.output_activation == 'identity'):
            return R2(z,p)
        
        return accuracy(z,p)
    
    def predict(self,X):
        '''
        Feeds input forward to produce
        output of network.
        Predicts output based on X.
        '''
        p = self._fast_feed_forward(X)
        if(self.output_activation=='linear'):
            return p
        return np.where(p<0.5,0,1)
    
    def predict_continuous(self,X):
        '''
        Outputs the prediction with continuous
        values. AKA predicts probabilities.
        '''
        return self._fast_feed_forward(X)
                     
    def _init_w(self,n_features,n_neurons):
        if (self.w_init=='normal'):
            return random.randn(n_features, n_neurons)
        elif(self.w_init=='uniform'):
            return random.uniform(0,1,(n_features, n_neurons))
        elif(self.w_init =='glorot'):
            limit = np.sqrt(6.0/(n_features+n_neurons))
            return random.uniform(-limit,limit,(n_features, n_neurons))
    
    def _init_b(self,lenght):
        return full(lenght,self.b_init)
    
    def _feed_forward(self,X):
        '''
        Feeds X forward in the network and 
        stores each activations in the layers
        '''
        a = X
        for layer in self.layers:
            #Weighted sumjh
            layer.z_h = matmul(a,layer.weights) + layer.bias
            #Store activations in layer object
            layer.activations = layer.activate(layer.z_h)
            #Set activation as input for next iteration
            a = layer.activations

        return a
    
    def _fast_feed_forward(self,X):    
        '''
        Feeds X forward in the network and
        returns output layer activation
        without storing intermitten activations
        '''             
        a = X
        for layer in self.layers:
            #Weighted sum
            z_h = matmul(a,layer.weights) + layer.bias            
            #Set activation as input for next iteration
            a = layer.activate(z_h)
            
        return a
    
    def _backpropagation(self,X,z):
        '''
        Performs backpropagation across the entire
        network.
        This function updates
        ''' 
        #Do last layer first
        error = self.layers[-1].activations - z
        weights = self.layers[-1].weights #Store weights for next layer
        w_grad, b_grad = self._calc_grads(weights,self.layers[-2].activations, error)
        self._update_weights_and_biases(self.layers[-1],w_grad,b_grad)
        
        #Do hidden layers
        for i in range(len(self.layers)-2,0,-1):
            #Next layer error
            error = (error @ weights.T)*self.layers[i].derivative(self.layers[i].z_h,error)     
            weights = self.layers[i].weights #Store weights for next iteration
            w_grad, b_grad = self._calc_grads(weights,self.layers[i-1].activations, error) 
            #Back propagate one step
            self._update_weights_and_biases(self.layers[i],w_grad,b_grad)
                 
        #Do first layer
        error = matmul(error, weights.T)*self.layers[0].derivative(self.layers[0].z_h,error)
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
        '''
        Calculates the gradients for current_layer's
        weights and biases.
        '''
        w_grad = matmul(prev_layer_a.T,error)
        b_grad = np.mean(error,axis=0)
        #Add l2 regularization if specified
        if(self.regularization=='l2'):
            w_grad += self.lmb * current_layer_w
        
        w_grad /=self.batch_size       
        return w_grad,b_grad

 

