'''
Created on Mar 26, 2013

@author: sgangireddy
'''
'''
Created on 21 Feb 2013

@author: s1264845
'''
import theano
import theano.tensor as T
import time, numpy
from vocab_create import Vocabulary
from data_provider_modified import DataProvider
import os, sys
from learn_rates import LearningRateNewBob, LearningRateList
from learn_rates import LearningRate
from logistic_regression import LogisticRegression
from mlp_new_uni import MLP
from mlp_new_uni import HiddenLayer
from cache import TNetsCacheSimple, TNetsCacheLastElem
from numpy.core.numeric import dtype
from utils import GlobalCfg
from mlp_save import save_mlp, save_posteriors
import math
from train_model import training
from test_model import testing        
def train_mlp(L1_reg = 0.0, L2_reg = 0.0000, num_batches_per_bunch = 512, batch_size = 1, num_bunches_queue = 5, offset = 0, path_name = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/s1264845/data/'):
    

    voc_list = Vocabulary(path_name + 'train')
    voc_list.vocab_create()
    vocab = voc_list.vocab
    vocab_size = voc_list.vocab_size
    
    voc_list_valid = Vocabulary(path_name + 'valid')
    voc_list_valid.vocab_create()
    valid_words_count = voc_list_valid.count
    #print valid_words_count
    valid_lines_count = voc_list_valid.line_count
    #print valid_lines_count

    voc_list_test = Vocabulary(path_name + 'test')
    voc_list_test.vocab_create()
    test_words_count = voc_list_test.count
    #print test_words_count
    test_lines_count = voc_list_test.line_count
    #print test_lines_count
 
    dataprovider_train = DataProvider(path_name + 'train', vocab, vocab_size )
    dataprovider_valid = DataProvider(path_name + 'valid', vocab, vocab_size )
    dataprovider_test = DataProvider(path_name + 'test', vocab, vocab_size )

    #exp_name = 'fine_tuning.hdf5'
    
    print '..building the model'

    #symbolic variables for input, target vector and batch index
    index = T.lscalar('index')
    x = T.fvector('x')
    y = T.ivector('y')
    learning_rate = T.fscalar('learning_rate') 

    #theano shared variables for train, valid and test
    train_set_x = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    valid_set_x = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    test_set_x = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    rng = numpy.random.RandomState() 
   
    classifier = MLP(rng = rng, input = x, n_in = vocab_size, fea_dim = 30, context_size = 2, n_hidden = 60 , n_out = vocab_size)

    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    
    #constructor for learning rate class
    learnrate_schedular = LearningRateNewBob(start_rate=0.005, scale_by=.5, max_epochs=9999,\
                                    min_derror_ramp_start=.01, min_derror_stop=.01, init_error=100.)

    #learnrate_schedular = LearningRateList(learn_list)

    frame_error = classifier.errors(y)
    likelihood = classifier.sum(y)
    
    #test_model
    test_model = theano.function(inputs = [], outputs = likelihood,  \
                                 givens = {x: test_set_x,
                                           y: test_set_y})
    #validation_model
    validate_model = theano.function(inputs = [], outputs = [frame_error, likelihood], \
                                     givens = {x: valid_set_x,
                                               y: valid_set_y})

    gradient_param = []
    #calculates the gradient of cost with respect to parameters 
    for param in classifier.params:
        gradient_param.append(T.cast(T.grad(cost, param), 'float32'))
        
    updates = []
    #updates the parameters
    for param, gradient in zip(classifier.params, gradient_param):
        updates.append((param, param - learning_rate * gradient))
    
    #training_model
    train_model = theano.function(inputs = [theano.Param(learning_rate, default = 0.01)], outputs = cost, updates = updates, \
                                 givens = {x: train_set_x,
                                           y: train_set_y})
   

    training(dataprovider_train, dataprovider_valid, learnrate_schedular, classifier, train_model, validate_model, train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size, num_batches_per_bunch, valid_words_count, valid_lines_count) 
    testing(dataprovider_test, classifier, test_model, test_set_x, test_set_y, test_words_count, test_lines_count)
train_mlp()
