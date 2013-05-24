'''
Created on May 14, 2013

@author: sgangireddy
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
        
        self.output_linear = T.dot(self.input, self.W) + self.b
        if activation != None:
            self.output = activation(T.dot(self.input, self.W) + self.b)
        else:
            self.output = self.output_linear

        # parameters of the model and deltas
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]
        
class RNN_hiddenlayer(object):
    
    def __init__(self, rng, input3, initial_hidden, n_in, n_hidden):
        self.input3 = input3
        self.initial_hidden = initial_hidden
                        
        matrix1 = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_in + n_hidden)),
                high = numpy.sqrt(6./(n_in + n_hidden)),
                size = (n_in, n_hidden)), dtype = 'float32')
        
        self.W1 = theano.shared(value = matrix1, name = 'W1')
        
        matrix2 = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_hidden + n_hidden)),
                high = numpy.sqrt(6./(n_hidden + n_hidden)),
                size = (n_hidden, n_hidden)), dtype = 'float32')
        
        self.W2 = theano.shared(value = matrix2, name = 'W2')
        
        b_values = numpy.zeros((n_hidden,), dtype= 'float32')
        
        self.b = theano.shared(value = b_values, name ='b')
        
        #self.intial_hidden = theano.shared(numpy.zeros(n_hidden, ), dtype = 'float32', name = 'intial_hidden')
        
        self.output = T.tanh( T.add(T.add(T.dot(self.input3, self.W1), T.dot(self.initial_hidden, self.W2)), self.b))
        
        self.params = [self.W2, self.b, self.W1]
        
class ProjectionLayer(object):
        
    def __init__(self, rng, input1, input2, n_in, fea_dim):
        
        self.input1 = input1
        self.input2 = input2
        #mean= numpy.zeros(fea_dim, dtype = 'float32')
        #cov = numpy.identity(fea_dim, dtype = 'float32')
        #feature_values = numpy.asarray(rng.multivariate_normal(mean, cov, n_in), dtype = 'float32')
        feature_values = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_in + fea_dim)),
                high = numpy.sqrt(6./(n_in + fea_dim)),
                size = (n_in, fea_dim)), dtype = 'float32')
        
        self.W = theano.shared(value = feature_values, name = 'W')
        
        self.output1 = T.dot(self.input1, self.W)
        self.output2 = T.dot(self.input2, self.W)
        self.output = T.concatenate([self.output1, self.output2])
              
        self.params = [self.W]
        
class OutputLayer(object):
    
    def __init__(self, rng, input, n_in, n_out):
        self.input = input
        
        matrix = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_in + n_out)),
                high = numpy.sqrt(6./(n_in + n_out)),
                size = (n_in, n_out)), dtype = 'float32')
        self.W = theano.shared(value = matrix, name = 'W')
        
        self.output = T.dot(self.input, self.W)
        
        self.params = [self.W]
        
class Softmax(object):
    
    def __init__(self, input, n_out):
        
        self.input = input
        
        b_values = numpy.zeros((n_out,), dtype = 'float32')
        self.b = theano.shared(value = b_values, name = 'b')
        
        self.p_y_given_x = T.nnet.softmax(self.input + self.b)
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)
        
        self.params = [self.b]
    
    def negative_loglikelihood_sum(self, y):
        return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    def negative_loglikelihood_values(self, y):
    #return T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]
        return T.cast(self.p_y_given_x[T.arange(y.shape[0]),y], dtype = 'float32')
    
    def negative_loglikelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y], dtype = 'float32')

        
class MLP_RNN(object):
    
    def __init__(self, rng, input1, input2, input3, initial_hidden, n_in, fea_dim, context_size, n_hidden, n_out, \
                  W_hid=None, b_hid=None, W_out=None, b_out=None):
        
        
        if rng is None:
           rng = numpy.random.RandomState()
        
        self.MLPinputlayer = ProjectionLayer(rng, input1, input2, n_in, fea_dim)
        
        self.MLPhiddenlayer = HiddenLayer(rng = rng, input = self.MLPinputlayer.output, 
                                 n_in = fea_dim * context_size, n_out = n_hidden,
                                 W_values = W_hid, b_values = b_hid,
                                 activation = theano.tensor.tanh)
        self.RNNhiddenlayer = RNN_hiddenlayer(rng, input3, initial_hidden, n_in, n_hidden)
        
        self.MLPoutput = OutputLayer(rng, self.MLPhiddenlayer.output, n_hidden, n_out)

        self.RNNoutput = OutputLayer(rng, self.RNNhiddenlayer.output, n_hidden, n_out)
        
        self.output = self.MLPoutput.output + self.RNNoutput.output
        
        self.Softmaxoutput = Softmax(self.output, n_out)
        
        self.cost = self.Softmaxoutput.negative_loglikelihood
        
        self.sum = self.Softmaxoutput.negative_loglikelihood_sum
        
        self.likelihood = self.Softmaxoutput.negative_loglikelihood_values
        
        self.params = self.RNNoutput.params + self.Softmaxoutput.params + self.RNNhiddenlayer.params
        
        self.MLPparams = self.MLPoutput.params + self.Softmaxoutput.params + self.MLPhiddenlayer.params + self.MLPinputlayer.params

	self.no_of_layers = len(self.MLPparams)
