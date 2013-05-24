'''
Created on 21 Feb 2013

@author: s1264845
'''
import theano
import theano.tensor as T
import numpy
from logistic_regression import LogisticRegression

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, W_values = None, b_values = None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden 
                              layer
        """
        self.input = input
        if not self.input:
            self.input = T.fmatrix('input')
        
        # Both W and b (W_values and b_values respectively) may be already shared
        # as theano variable. It happens when we want to build an unrolled 
        # generative network prior to generative back-fitting and we share the parameters 
        # with DNN/DBN structures 
         
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype 
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you 
        #        should use 4 times larger initial weights for sigmoid 
        #        compared to tanh
        #        We have no info for other function, so we use the same as tanh.
        
        if W_values is None:
            W_values = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_in+n_out)),
                high = numpy.sqrt(6./(n_in+n_out)),
                size = (n_in, n_out)), dtype = 'float32')
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
        
        if b_values is None:
            b_values = numpy.zeros((n_out,), dtype= 'float32')
        
        if isinstance(W_values, theano.Variable):
            print 'W is theano.Variable'
            self.W = W_values
        else:
            self.W = theano.shared(value = W_values, name ='W')
            
        if isinstance(b_values, theano.Variable):
            print 'b is theano.Variable'
            self.b = b_values
        else:
            self.b = theano.shared(value = b_values, name ='b')
            
        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out), \
                                                        dtype = 'float32'), name='delta_W')
        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True), \
                                                        dtype = 'float32'), name='delta_b')
        
        #self.output_linear = T.dot(self.input, self.W) + self.b
        if activation != None:
            self.output = activation(T.dot(self.input, self.W) + self.b)
	    self.params = [self.W, self.b]
	    self.delta_params = [self.delta_W, self.delta_b]
        else:
            self.output = T.dot(self.input, self.W)
	    self.params = [self.W]
            self.delta_params = [self.delta_W]

        # parameters of the model and deltas
        #self.params = [self.W, self.b]
        #self.delta_params = [self.delta_W, self.delta_b]

       
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model 
    that has one layer or more of hidden units and nonlinear activations. 
    Intermediate layers usually have as activation function thanh or the 
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the 
    top layer is a softamx layer (defined here by a ``LogisticRegression`` 
    class). 
    """

    def __init__(self, rng, input, n_in, n_hidden1, n_hidden2, n_out, \
                  W_hid=None, b_hid=None, W_out=None, b_out=None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the 
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in 
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units 

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in 
        which the labels lie

        """
        
        if rng is None:
            rng = numpy.random.RandomState()

        # Since we are dealing with a one hidden layer MLP, this will 
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer 
        # implementing any other nonlinearity
        self.hiddenLayer1 = HiddenLayer(rng = rng, input = input, 
                                 n_in = n_in, n_out = n_hidden1,
                                 W_values = W_hid, b_values = b_hid,
                                 activation = None)
        
        self.hiddenLayer2 = HiddenLayer(rng = rng, input = self.hiddenLayer1.output, 
                                 n_in = n_hidden1, n_out = n_hidden2,
                                 W_values = W_hid, b_values = b_hid,
                                 activation = theano.tensor.tanh)

        # The logistic regression layer gets as input the hidden units 
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(rng = rng, input = self.hiddenLayer2.output, n_in  = n_hidden2, n_out = n_out, W_values = None, b_values = None)

        # L1 norm ; one regularization option is to enforce L1 norm to 
        # be small 
        self.L1 = abs(self.hiddenLayer1.W).sum() \
                + abs(self.logRegressionLayer.W).sum() + abs(self.hiddenLayer2.W).sum()

        # square of L2 norm ; one regularization option is to enforce 
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer1.W**2).sum() \
                    + (self.logRegressionLayer.W**2).sum() #+ (self.hiddenLayer2.W**2).sum()

        # negative log likelihood of the MLP is given by the negative 
        # log likelihood of the output of the model, computed in the 
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

	self.sum = self.logRegressionLayer.negative_log_likelihood_sum

        # takes logs of the last softmax layer
        self.log_posteriors = self.logRegressionLayer.log_posteriors
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        # returns the labels and predictions
        self.log_error_results = self.logRegressionLayer.log_error_results
        
        self.cost =  self.negative_log_likelihood
        
        #self.delta_params = self.hiddenLayer.delta_params + self.logRegressionLayer.delta_params
        self.params = self.logRegressionLayer.params + self.hiddenLayer2.params + self.hiddenLayer1.params
        self.delta_params = self.hiddenLayer1.delta_params + self.hiddenLayer2.delta_params + self.logRegressionLayer.delta_params
	self.no_of_layers = len(self.params)
