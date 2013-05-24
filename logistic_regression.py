'''
Created on 21 Feb 2013

@author: s1264845
'''
import theano
import theano.tensor as T
import numpy
from numpy.core.numeric import dtype

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W` 
    and bias vector :math:`b`. Classification is done by projecting data 
    points onto a set of hyperplanes, the distance to which is used to 
    determine a class membership probability. 
    """
    
    def __init__(self, rng, input, n_in, n_out, W_values = None, b_values = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the 
                      architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in 
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in 
                      which the labels lie

        """ 
        
        self.input = input
        if self.input is None:
            self.input = T.fmatrix('input')
               
        if W_values is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out) 
            # W_values = numpy.zeros((n_in, n_out), dtype = 'float32')
            rng = numpy.random.RandomState()
           # W_values = numpy.zeros((n_in, n_out), dtype = 'float32')
            W_values = numpy.asarray(rng.uniform(
                low  = -numpy.sqrt( 6. / ( n_in + n_out ) ),
                high = numpy.sqrt(6. / ( n_in + n_out ) ),
                size = (n_in, n_out)), dtype = 'float32')
        
        if b_values is None:
            # initialize the baises b as a vector of n_out 0s
            b_values = numpy.zeros((n_out,), dtype = 'float32')
        
        self.W = theano.shared(value = W_values, name = 'W')
        self.b = theano.shared(value = b_values, name = 'b')
        
        #self.delta_W = theano.shared(value = numpy.zeros((n_in, n_out)), name = 'delta_W', dtype = 'float32')
        #self.delta_b = theano.shared(value = numpy.zeros((n_out,)), name = 'delta_b', dtype = 'float32')
        
        self.delta_W = theano.shared(value = numpy.zeros((n_in,n_out), \
                                                        dtype = 'float32'), name='delta_W')
        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True), \
                                                        dtype = 'float32'), name='delta_b')
    
        #self.priors = None    
        #if class_counts != None:
        #    assert class_counts.shape[0] == n_out
        #    class_counts_priors = numpy.asarray(class_counts/float(class_counts.sum()), dtype=theano.config.floatX)
        #    self.priors = theano.shared(value = class_counts_priors, name = 'priors')
        
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
	#print self.p_y_given_x

            
        #if self.priors!=None:
        #    self.softmax_activations = self.p_y_given_x/self.priors
        #    self.softmax_log_activations = T.log(self.p_y_given_x) - T.log(self.priors)
        #    self.linear_activations = (T.dot(self.input, self.W) + self.b)
        #else:
        #    self.softmax_activations = self.p_y_given_x
        #    self.softmax_log_activations = T.log(self.p_y_given_x)
        #    self.linear_activations = (T.dot(self.input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the    
                  correct label

        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and one column per class 
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        # and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        #
        # NOTE: sum, given the learning rate is right for a mini-batch may give better results
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y], dtype = 'float32')
	#return T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]

    def negative_log_likelihood_values(self, y):
	#return T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]
	    return T.cast(self.p_y_given_x[T.arange(y.shape[0]),y], dtype = 'float32')
    
    def likelihood_values(self, y):
        return T.cast(self.p_y_given_x[T.arange(y.shape[0]),y], dtype = 'float32')
 
    def negative_log_likelihood_sum(self, y):
        return T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    def mean_sqare_error(self, y_err):
        return 0.5*T.mean(T.sqr(self.p_y_given_x - y_err))
    
    def cross_entropy(self, y_err):
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y_err))
    
    def error_lost_function(self, y_err):
        return T.sum(T.sum(T.sqr(self.p_y_given_x - y_err), axis=1))

    def log_posteriors(self):
        return self.output_activations
    
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch 
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the 
                  correct label
        """
        
        # check if y has same dimension of y_pred 
        if y.ndim != self.y_pred.ndim:
	    #print y.ndim, self.y_pred.ndim
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y), dtype = 'float32')
        else:
            raise NotImplementedError()

    def log_error_results(self, y):
        """Returns a matrix [reference labels; predicted labels] for debugging purposes
        """
        # check if y has same dimension of y_pred 
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return [self.y_pred, y]
        else:
            raise NotImplementedError()
