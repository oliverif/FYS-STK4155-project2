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
    '''
    Neural Network with SGD optimizer.
    
    This model minimizes either the squared error
    loss function or the binary cross entropy loss function
    to obtain optimal weights and biases.
    
    The class inherits from SGD_optimizer as it is the
    base class of all models.
    
    Parameters:
    ----------
    hidden_layer_sizes: tuple, default=(50,)
        Defines the architecture of the network. Each entry in the tuple
        is a layer. The number is the amount of neurons for that layer.
        E.g. (50,100) has two hidden layers, first with 50 neurons,
        and second with 100.
        
    hidden_activation: str{'sigmoid','relu','leakyrelu'}, default='relu'
        The activation function to use in the hidden layers.
        
    output_activation: str{'sigmoid','identity'}
        The activation function to use for the output. Sigmoid is used
        for classification problems while identity is used for regression problems.
        
    n_categories: int, default=1
        The number of output categories. Only useful for multiclass. Must be 1
        when nn is regressor.
        
    w_init: str{'normal','uniform','glorot'},default='glorot
        The initialization scheme to use when initializing the weights of the
        layers.
        
    b_init: float, default=0.001
        The initialization value to give all biases. All biases are initialized
        to constant values in this model.
        
    loss_func: str{'cross_entropy','squared_loss'} default='cross_entropy'
        The loss function to use. Cross entropy is often used in classification
        problems while squared loss is used in regression.
               
    regularization: str{'l2',None} default='l2'
        Wether or not to use l2 regularization
        
    lmb: float, default = 0.001
        The regularization parameter. Only used if regularization is 'l2'
        
    momentum: float, default = 0.5
        The portion of the former velocity to influde in new velocity calculation.
        
    schedule: str{'constant','invscaling','decaying'}, default='constant'
        The learning rate schedule.
        
    lr0: float, default=0.01
        The inital learning rate. Only used by constant and invscaling.
        
    batch_size: int, default=32
        The amount of data points in each batch during mini batch stochastic
        gradient descent. Should be a multiplum of 2.
        
    n_epochs: int, default=100
        Number of epochs to train.
        
    t0: int, default=50
        parameter in the decaying schedule
    
    t1: int, default=300
        parameter in the decaying schedule
        
    power_t: float, default=0.05
        parameter in the invscaling schedule
        
    val_fraction: float, default=0.1
        The portion of the input data to set aside as validation data.
        Validation data is used to study loss and score during training.
    '''
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
        
        A layer's weight matrix has the shape of 
        (prev_n_neurons,current_n_neurons). I.e the 
        previous layer's weight matrix must be defined
        before the current can.
        The first hidden layer will have shape
        (n_features, current_layer_neurons).
        
        Inputs:
        -------
        input_shape: tuple
            The shape of the input data so that
            all subsequent layers can be initialized.
        '''
        
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
        
        This function first performs feed forward
        to generate the activations. The loss is then
        capturen, and then backpropagation is performed
        updating the weights.
        
        If regularization is employed, this is added
        to the loss as well.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data
            
        Output:
        -------
        loss: float
            The loss captured right before parameter
            update.
        '''
        #Feed foward, i.e predict
        p = self._feed_forward(X)
        #Capture loss
        loss = self.loss_func(z,p)
        #Backpropagation
        self._backpropagation(X,z)
        
        #Add regularization
        if(self.regularization == 'l2'):
            for layer in self.layers:
                w = layer.weights.ravel()  
                # Dividing by n_samples in current batch makes 
                # regularization comparable across several batch 
                # sizes. 
                loss += self.lmb * (w @ w) / (2*z.shape[0])            
        return loss
    
    def score(self,X,z):
        '''
        Returns the coefficient of determination(R2) for the prediction if
        NN is regressor.
        Returns the mean accuracy of the prediction if NN is classifier.
        
        This function predicts based on X and compares its prediction with
        z.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data  
        '''
        #Predict output
        p = self.predict(X)
        #Ensure correct shape
        if(len(z.shape)==1):
            z = z.reshape(-1,1)
        #MLP is regressor        
        if (self.output_activation == 'identity'):
            return R2(z,p)
        #MLP is classifier 
        return accuracy(z,p)
    
    def predict(self,X):
        '''
        Feeds input forward to produce
        output of network.
        Predicts output based on X.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        '''
        p = self._fast_feed_forward(X)
        #Continous if regressor
        if(self.output_activation=='identity'):
            return p
        #Binary if classifier. Threshold is
        #set to 0.5 in this case. I.e
        #above 50% probablity = 1.
        return np.where(p<0.5,0,1)
    
    def predict_continuous(self,X):
        '''
        Outputs the prediction with continuous
        values. AKA predicts probabilities.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        '''
        return self._fast_feed_forward(X)
                     
    def _init_w(self,n_features,n_neurons):
        '''
        Returns a numpy array with distributed
        values according to self.w_init.
        
        This function is used to initialize the 
        weights of the neural network.
        
        Inputs:
        -------
        n_features: int
            number of features or input
            nodes to the layer
            
        n_neurons: int
            Amount of neurons for the current
            layer.
        '''
        #Normally distribute weight values
        if (self.w_init=='normal'):
            return random.randn(n_features, n_neurons)
        #Uniformly distribute weight values
        elif(self.w_init=='uniform'):
            return random.uniform(-1,1,(n_features, n_neurons))
        #Uniformly distribute weight values 
        #with min max according to Glorot et al.
        elif(self.w_init =='glorot'):
            limit = np.sqrt(6.0/(n_features+n_neurons))
            return random.uniform(-limit,limit,(n_features, n_neurons))
    
    def _init_b(self,lenght):
        '''
        Returns a numpy array containing the
        same values.
        
        This function is used to initialize the 
        biases to some constant defined by self.b_init 
        given in __init__.
        
        Inputs:
        -------
        length: int
            The length of the bias array, i.e amount of nodes
            in layer.
            
        Outpus:
        -------
        ndarray(length,)
            Array containing self.b_init value
        ''' 
        return full(lenght,self.b_init)
    
    def _feed_forward(self,X):
        '''
        Feeds X forward in the network and 
        stores each of the activations in the layers
        
        This function mainly consist of a loop
        iterating the layers from first to last.
        Here it calculates the weightet sum of
        all incoming activations and adds the bias.
        These values are then activated through
        the activation function. Lastly a is set
        to this activation so that the next layer can
        use it.
        
        Note that for the very first layer, the
        activations is simply X.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        '''
        #Initial "activation", i.e input nodes
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
        
        This function calculates error and gradients for
        each layer staring at the outputlayer and working
        it's way backwards. As it loops backwards it also
        updates the parameters.
        
        Note that error and gradient are both calculated
        before moving to next layer. A common method
        is to first calculate error for all layers,
        and then calculate gradients for all, however for
        simplicity in the for loop error and gradients
        are calculated pairwise instead.
        
        The first step is to update the last layer as the error
        is calculated slightly differently here.
        After this the function enters a loop iterating all the
        hidden layers as they have the same methods for
        calculating error and gradient, and updating parameters.
        After the loop the first layer is updated as it
        uses X to calculate it's gradient.
        
        Inputs:
        -------
        X: ndarray(n_samples,n_features)
            Design matrix
        
        z: ndarray(n_samples,1)
            Target data  
        ''' 
        #Do last layer first
        #Last layer error
        error = self.layers[-1].activations - z
        #Store weights for next layer
        weights = self.layers[-1].weights 
        #Calculate the gradients
        w_grad, b_grad = self._calc_grads(weights,self.layers[-2].activations, error)
        #Update layer accordingly
        self._update_weights_and_biases(self.layers[-1],w_grad,b_grad)
        
        #Do hidden layers
        for i in range(len(self.layers)-2,0,-1):
            #Next layer error
            error = (error @ weights.T)*self.layers[i].derivative(self.layers[i].z_h,error) 
            #Store weights for next layer    
            weights = self.layers[i].weights
            #Calculate the gradients
            w_grad, b_grad = self._calc_grads(weights,self.layers[i-1].activations, error) 
            #Update layer accordingly
            self._update_weights_and_biases(self.layers[i],w_grad,b_grad)
                 
        #Do first layer
        #First layer error
        error = matmul(error, weights.T)*self.layers[0].derivative(self.layers[0].z_h,error)
        #Calculate the gradients
        w_grad, b_grad = self._calc_grads(self.layers[0].weights, X, error)
        #Update layer accordingly
        self._update_weights_and_biases(self.layers[0],w_grad,b_grad)
       
    def _update_weights_and_biases(self,layer,w_grad,b_grad):
        '''
        Updates the weights and biases of layer.
        Either updates with momentum or simply
        subtracts w_grad and b_grad from
        weights and biases respectively.
        '''
        
        if (self.momentum): 
            #Calculate and store velocities for weights and bias gradients   
            layer.v_w = self.momentum*layer.v_w - self.lr*w_grad
            layer.v_b = self.momentum*layer.v_b - self.lr*b_grad
            #Update weights and biases with velocity
            layer.weights += layer.v_w
            layer.bias += layer.v_b
            
        else:
            #Update weights and biases with gradients
            layer.weights -=  self.lr*w_grad
            layer.bias -=  self.lr*b_grad
             
    def _calc_grads(self,current_layer_w,prev_layer_a, error):
        '''
        Calculates the gradients for current_layer's
        weights and biases.
        '''
        #Weight gradient is a function of previous
        #layer activation and current layer error.
        w_grad = matmul(prev_layer_a.T,error)
        #Bias gradient
        b_grad = np.mean(error,axis=0)
        #Add l2 regularization if specified
        if(self.regularization=='l2'):
            w_grad += self.lmb * current_layer_w
        #Dividing by batch size allows for more comparable results with
        #different batch sizes.
        w_grad /=self.batch_size       
        return w_grad,b_grad

 

