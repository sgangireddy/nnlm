'''
Created on May 13, 2013

@author: sgangireddy
'''
import theano
import theano.tensor as T
import numpy
from logistic_regression import LogisticRegression
            
class RNN_InputLayer(object):
    
    def __init__(self, rng, input, intial_hidden, n_in, n_hidden):
        self.input = input
        self.intial_hidden = intial_hidden
                        
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
        
        self.output = T.nnet.sigmoid(T.add(T.dot(self.input, self.W1), T.dot(self.intial_hidden, self.W2)))
        
        self.params = [self.W2, self.W1]
        
class Softmax(object):
    def __init__(self, rng, input, n_in, n_out):
        self.input = input
        
        W_values = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_in + n_out)),
                high = numpy.sqrt(6./(n_in + n_out)),
                size = (n_in, n_out)), dtype = 'float32')
        
        self.W = theano.shared(value = W_values, name = 'W')

        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W))
        
        self.params = [self.W]
        
    def negative_loglikelihood_sum(self, y):
        return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y], dtype = 'float32')
    
    def negative_loglikelihood_values(self, y):
        return T.cast(self.p_y_given_x[T.arange(y.shape[0]),y], dtype = 'float32')
    
    def negative_loglikelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y], dtype = 'float32')

        

class RNN(object):

    def __init__(self, rng, input, intial_hidden, n_in, n_hidden, n_out, \
                 W_out=None, b_out=None):
        
        if rng is None:
            rng = numpy.random.RandomState()
        
        self.inputlayer = RNN_InputLayer(rng, input, intial_hidden, n_in, n_hidden)
                    
        self.SoftmaxLayer = Softmax(rng, self.inputlayer.output, n_hidden, n_out)
        
        self.negative_log_likelihood = self.SoftmaxLayer.negative_loglikelihood

        self.sum = self.SoftmaxLayer.negative_loglikelihood_sum
        
        self.likelihood = self.SoftmaxLayer.negative_loglikelihood_values
        
        self.params = self.SoftmaxLayer.params + self.inputlayer.params
        
        self.no_of_layers = len(self.params)
