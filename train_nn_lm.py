'''
Created on Mar 26, 2013

@author: sgangireddy
'''
#import data_provider
'''
Created on 21 Feb 2013

@author: s1264845
'''
import theano
import theano.tensor as T
import time, numpy
from vocab_create import Vocabulary
from data_provider_modified import DataProvider
import os, sys, getopt
from learn_rates import LearningRateNewBob, LearningRateList
from learn_rates import LearningRate
from logistic_regression import LogisticRegression
from mlp_new_uni import MLP
from mlp_new_uni import HiddenLayer
#from cache import TNetsCacheSimple, TNetsCacheLastElem
from numpy.core.numeric import dtype
import h5py
from mlp_save import save_mlp, save_posteriors, save_learningrate
from vocab_hash import Vocabularyhash
      
def train_mlp(feature_dimension, context, hidden_size, weight_path, file_name1, file_name2, file_name3, L1_reg = 0.0, L2_reg = 0.0000, path_name = '/exports/work/inf_hcrc_cstr_udialogue/siva/data/'):
    

    #voc_list = Vocabulary(path_name + 'train_modified1')
    #voc_list.vocab_create()
    #vocab = voc_list.vocab
    #vocab_size = voc_list.vocab_size
    #short_list = voc_list.short_list
    #short_list_size = voc_list.short_list_size
    #path = '/exports/work/inf_hcrc_cstr_udialogue/siva/data_normalization/vocab/wlist5c.nvp'
    voc_list = Vocabularyhash('/exports/work/inf_hcrc_cstr_udialogue/siva/data_normalization/vocab/wlist5c.nvp')
    voc_list.hash_create()
    vocab = voc_list.voc_hash
    vocab_size = voc_list.vocab_size
    
    #dataprovider_train = DataProvider(path_name + 'train', vocab, vocab_size, short_list )
    #dataprovider_valid = DataProvider(path_name + 'valid', vocab, vocab_size, short_list )
    #dataprovider_test = DataProvider(path_name + 'test', vocab, vocab_size , short_list)
    
    dataprovider_train = DataProvider(path_name + 'train_modified1_20m', vocab, vocab_size)
    dataprovider_valid = DataProvider(path_name + 'valid_modified1', vocab, vocab_size)
    dataprovider_test = DataProvider(path_name + 'test_modified1', vocab, vocab_size)

    print '..building the model'

    #symbolic variables for input, target vector and batch index
    index = T.lscalar('index')
    x1 = T.fvector('x1')
    x2 = T.fvector('x2')
    y = T.ivector('y')
    learning_rate = T.fscalar('learning_rate') 

    #theano shared variables for train, valid and test
    train_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    valid_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    test_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    rng = numpy.random.RandomState() 
   
    classifier = MLP(rng = rng, input1 = x1, input2 = x2,  n_in = vocab_size, fea_dim = int(feature_dimension), context_size = int(context), n_hidden =int(hidden_size), n_out = vocab_size)
    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    
    #constructor for learning rate class
    learnrate_schedular = LearningRateNewBob(start_rate=0.005, scale_by=.5, max_epochs=9999,\
                                    min_derror_ramp_start=.01, min_derror_stop=.01, init_error=100.)

    frame_error = classifier.errors(y)
    log_likelihood = classifier.sum(y)
    likelihood = classifier.likelihood(y)
    
    #test_model
    test_model = theano.function(inputs = [], outputs = [log_likelihood, likelihood],  \
                                 givens = {x1: test_set_x1,
                                           x2: test_set_x2,
                                           y: test_set_y})
    #validation_model
    validate_model = theano.function(inputs = [], outputs = [frame_error, log_likelihood], \
                                     givens = {x1: valid_set_x1,
                                               x2: valid_set_x2,
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
    train_model = theano.function(inputs = [learning_rate], outputs = [cost], updates = updates, \
                                 givens = {x1: train_set_x1,
                                           x2: train_set_x2,
                                           y: train_set_y})


    print '.....training'
    best_valid_loss = numpy.inf
    start_time = time.time()
    while(learnrate_schedular.get_rate() != 0):
    
        print 'learning_rate:', learnrate_schedular.get_rate()		
        print 'epoch_number:', learnrate_schedular.epoch        
        frames_showed, progress = 0, 0
        start_epoch_time = time.time()
        dataprovider_train.reset()
        
	for feats_lab_tuple in dataprovider_train:

            features, labels = feats_lab_tuple 
            
            if labels is None or features is None:
                continue                             
            frames_showed += features.shape[0]
            for temp, i in zip(features, xrange(len(labels))):
                temp_features1 = numpy.zeros(vocab_size, dtype = 'float32')
                temp_features2 = numpy.zeros(vocab_size, dtype = 'float32')
                temp_features1[temp[0]] = 1
                temp_features2[temp[1]] = 1
                train_set_x1.set_value(numpy.asarray(temp_features1, dtype = 'float32'), borrow = True)
                train_set_x2.set_value(numpy.asarray(temp_features2, dtype = 'float32'), borrow = True)
                train_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
                out = train_model(numpy.array(learnrate_schedular.get_rate(), dtype = 'float32'))
            progress += 1
            if progress%10000==0:
                end_time_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
            train_set_x1.set_value(numpy.empty((1), dtype = 'float32'))
            train_set_x2.set_value(numpy.empty((1), dtype = 'float32'))
            train_set_y.set_value(numpy.empty((1), dtype = 'int32'))
        
        end_time_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
        classifier_name = 'MLP' + str(learnrate_schedular.epoch)
        save_mlp(classifier, weight_path+file_name1 , classifier_name)
	save_learningrate(learnrate_schedular.get_rate(), weight_path+file_name3, classifier_name)
    
        print 'Validating...'
        valid_losses = []
        log_likelihood = []
        valid_frames_showed, progress = 0, 0
        start_valid_time = time.time() # it is also stop of training time
        dataprovider_valid.reset()
        
	for feats_lab_tuple in dataprovider_valid:            
            features, labels = feats_lab_tuple 
            
            if labels is None or features is None:
                continue                             
            valid_frames_showed += features.shape[0]                
            for temp, i in zip(features, xrange(len(labels))):
                temp_features1 = numpy.zeros(vocab_size, dtype = 'float32')
                temp_features2 = numpy.zeros(vocab_size, dtype = 'float32')
                temp_features1[temp[0]] = 1
                temp_features2[temp[1]] = 1
                valid_set_x1.set_value(numpy.asarray(temp_features1, dtype = 'float32'), borrow = True)
                valid_set_x2.set_value(numpy.asarray(temp_features2, dtype = 'float32'), borrow = True)
                valid_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
                out = validate_model()
                error_rate = out[0]
                likelihoods = out[1] 
                valid_losses.append(error_rate)
                log_likelihood.append(likelihoods)
            valid_set_x1.set_value(numpy.empty((1), 'float32'))
            valid_set_x2.set_value(numpy.empty((1), 'float32'))
            valid_set_y.set_value(numpy.empty((1), 'int32'))

            progress += 1
            if progress%1000==0:
                end_time_valid_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)
        
        end_time_valid_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)            
        this_validation_loss = numpy.mean(valid_losses)
        entropy = (-numpy.sum(log_likelihood)/valid_frames_showed)
        print this_validation_loss, entropy, numpy.sum(log_likelihood)
        
	if entropy < best_valid_loss:
           learning_rate = learnrate_schedular.get_next_rate(entropy)
           best_valid_loss = entropy
        else:
           learnrate_schedular.rate = 0.0
    end_time = time.time()
    print 'The fine tuning ran for %.2fm' %((end_time-start_time)/60.)
    
    print 'Testing...'
    log_likelihood = []
    likelihoods = []
    test_frames_showed, progress = 0, 0
    start_test_time = time.time() # it is also stop of training time
    dataprovider_test.reset()
    
    for feats_lab_tuple in dataprovider_test:
        
        features, labels = feats_lab_tuple 
            
        if labels is None or features is None:
            continue                             

        test_frames_showed += features.shape[0]                
        for temp, i in zip(features, xrange(len(labels))):
            temp_features1 = numpy.zeros(vocab_size, dtype = 'float32')
            temp_features2 = numpy.zeros(vocab_size, dtype = 'float32')
            temp_features1[temp[0]] = 1
            temp_features2[temp[1]] = 1
            test_set_x1.set_value(numpy.asarray(temp_features1, dtype = 'float32'), borrow = True)
            test_set_x2.set_value(numpy.asarray(temp_features2, dtype = 'float32'), borrow = True)
            test_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
            out = test_model()
            log_likelihood.append(out[0])
            likelihoods.append(out[1])
        progress += 1
        if progress%1000==0:
           end_time_test_progress = time.time()
           print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, test_frames_showed, end_time_test_progress - start_test_time)
    end_time_test_progress = time.time()
    print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                    %(progress, test_frames_showed, end_time_test_progress - start_test_time)            
    save_posteriors(log_likelihood, likelihoods, weight_path+file_name2)
    print numpy.sum(log_likelihood)
    likelihood_sum = (-numpy.sum(log_likelihood)/test_frames_showed)
    print 'entropy:', likelihood_sum

opts, extraparams = getopt.getopt(sys.argv[1:], "f:c:h:p:m:o:l:", ["--feature", "--context", "--hidden", "--path", "--file_name1", "--file_name2", "--file_name3"])
for o,p in opts:
  if o in ['-f','--feature']:
     feature_dimension = p
  elif o in ['-c', '--context']:
     context = p
  elif o in ['-h', '--hidden']:
     hidden_size = p
  #elif o in ['-d', '--hidden2']:
  #   hidden_size2 = p
  elif o in ['-p', '--path']:
     weight_path = p
  elif o in ['-m', '--file_name1']:
     file_name1 = p
  elif o in ['-o', '--file_name2']:
     file_name2 = p
  elif o in ['-l', '--file_name3']:
     file_name3 = p

train_mlp(feature_dimension, context, hidden_size, weight_path, file_name1, file_name2, file_name3)
