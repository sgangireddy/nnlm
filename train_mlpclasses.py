'''
Created on May 14, 2013

@author: sgangireddy
'''
import theano
import theano.tensor as T
import time, numpy
from vocab_class import Vocabulary
from data_provider_classes import DataProvider
import os, sys, getopt
from learn_rates import LearningRateNewBob, LearningRateList
from learn_rates import LearningRate
from logistic_regression import LogisticRegression
from mlp_classes1 import MLPClasses
from cache import TNetsCacheSimple, TNetsCacheLastElem
from numpy.core.numeric import dtype
from utils import GlobalCfg
#import h5py
#from mlp_save import save_mlp, save_posteriors, save_learningrate
#from vocab_hash import Vocabularyhash
      
def train_mlpclasses(path_name = '/exports/work/inf_hcrc_cstr_udialogue/siva/data/', n_hidden = int(sys.argv[1]), n_classes = int(sys.argv[2])): 

    voc_list = Vocabulary(path_name + 'train', n_classes)
    voc_list.vocab_create()
    voc_list.class_label()
    vocab = voc_list.vocab
    vocab_size = voc_list.vocab_size
    classes = voc_list.classes
    
    dataprovider_train = DataProvider(path_name + 'train', vocab, vocab_size, classes)
    dataprovider_valid = DataProvider(path_name + 'valid', vocab, vocab_size, classes)
    dataprovider_test = DataProvider(path_name + 'test', vocab, vocab_size, classes)
    
    print '..building the model'
    #symbolic variables for input, target vector and batch index
    index = T.lscalar('index')
    x1 = T.fvector('x1')
    x2 = T.fvector('x2')
    y_class = T.ivector('y_class')
    y_word = T.ivector('y_word')
    learning_rate = T.fscalar('learning_rate') 

    #theano shared variables for train, valid and test
    train_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_y_class = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    train_set_y_word = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    valid_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_y_class = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    valid_set_y_word = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    test_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_x3 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_y_class = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    test_set_y_word = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    rng = numpy.random.RandomState()
 
    classifier = MLPClasses(rng = rng, input1 = x1, input2 = x2, n_in = vocab_size, fea_dim = 50, context_size = 2, n_hidden = n_hidden, classes = classes)
    
    classcost = classifier.Classcost(y_class)
    wordcost = classifier.Wordcost(y_word)
        
    #constructor for learning rate class
    learnrate_schedular = LearningRateNewBob(start_rate = float(sys.argv[3]), scale_by = .5, max_epochs = 9999,\
                                    min_derror_ramp_start = .01, min_derror_stop = .01, init_error = 100.)

    class_log_likelihood = classifier.Classsum(y_class)
    word_log_likelihood = classifier.Wordsum(y_word)
    
    #test_model
    test_model = theano.function(inputs = [], outputs = [class_log_likelihood, word_log_likelihood],  \
                                 givens = {x1: test_set_x1,
                                           x2: test_set_x2,
                                           y_class: test_set_y_class,
                                           y_word: test_set_y_word})
    #validation_model
    validate_model = theano.function(inputs = [], outputs = [class_log_likelihood, word_log_likelihood], \
                                     givens = {x1: valid_set_x1,
                                               x2: valid_set_x2,
                                               y_class: valid_set_y_class,
                                               y_word: valid_set_y_word})

    gradient_wordparam = []
    gradient_classparam = []
    gradient_param = []
    #calculates the gradient of cost with respect to parameters 

    for param, i in zip(classifier.Classparams, xrange(len(classifier.Classparams))):
        if i <= 1:
            gradient_param.append(T.grad(classcost, param))
        else:
            gradient_classparam.append(T.grad(classcost, param))

    for param, i in zip(classifier.Wordparams, xrange(len(classifier.Wordparams))):
        if i <= 1:
            gradient_param.append(T.grad(wordcost, param))
        else:
            gradient_wordparam.append(T.grad(wordcost, param))
                
    for  i in xrange(len(gradient_wordparam)):
        gradient_param.append(gradient_classparam[i] + gradient_wordparam[i])
        
    updates = []
    #updates the parameters
    for param, gradient in zip(classifier.params, gradient_param):
        updates.append((param, param - learning_rate * gradient))

        #training_model
    train_model = theano.function(inputs = [learning_rate], outputs = [classcost, wordcost, classifier.WordoutputLayer.W, classifier.WordoutputLayer.b, class_log_likelihood, word_log_likelihood],\
                                 updates = updates,
                                 givens = {x1: train_set_x1,
                                           x2: train_set_x2,
                                           y_class: train_set_y_class,
                                           y_word: train_set_y_word})
    w_dict, b_dict = {}, {}
    
    for i in xrange(n_classes):
        W_values = numpy.asarray( rng.uniform(
                low  = - numpy.sqrt(6./(n_hidden + len(classes[i]))),
                high = numpy.sqrt(6./(n_hidden + len(classes[i]))),
                size = (n_hidden, len(classes[i]))), dtype = 'float32')
        w_dict[i] = W_values
        
        b_values = numpy.zeros((len(classes[i]), ), dtype = 'float32')
        b_dict[i]= b_values
     
    print '.....training'
    best_valid_loss = numpy.inf    
    start_time = time.time()
    while(learnrate_schedular.get_rate() != 0):
    
        print 'learning_rate:', learnrate_schedular.get_rate()
        print 'epoch_number:', learnrate_schedular.epoch    
        train_loglikelihood = []    
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
                train_set_x2.set_value(numpy.asarray(temp_features1, dtype = 'float32'), borrow = True)
                train_set_y_class.set_value(numpy.asarray([labels[i][1]], dtype = 'int32'), borrow = True)
                train_set_y_word.set_value(numpy.asarray([labels[i][0]], dtype = 'int32'), borrow = True)
                classifier.WordoutputLayer.W.set_value(numpy.asarray(w_dict[labels[i][1]], dtype = 'float32'), borrow = True)
                classifier.WordoutputLayer.b.set_value(numpy.asarray(b_dict[labels[i][1]], dtype = 'float32'), borrow = True)
                out = train_model(numpy.asarray(learnrate_schedular.get_rate(), dtype = 'float32'))
                w_dict[labels[i][1]],  b_dict[labels[i][1]] = out[2], out[3]
                train_loglikelihood.append(out[4]+out[5])
		#print out[4] + out[5]
            progress += 1
            if progress%10000==0:
                end_time_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
            train_set_x1.set_value(numpy.empty((1, ), dtype = 'float32'))
            train_set_x2.set_value(numpy.empty((1, ), dtype = 'float32'))
            train_set_y_class.set_value(numpy.empty((1), dtype = 'int32'))
            train_set_y_word.set_value(numpy.empty((1), dtype = 'int32'))
        
        end_time_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
	print numpy.sum(train_loglikelihood)
                          
        print 'Validating...'
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
                valid_set_y_class.set_value(numpy.asarray([labels[i][1]], dtype = 'int32'), borrow = True)
                valid_set_y_word.set_value(numpy.asarray([labels[i][0]], dtype = 'int32'), borrow = True)
                classifier.WordoutputLayer.W.set_value(numpy.asarray(w_dict[labels[i][1]], dtype = 'float32'), borrow = True)
                classifier.WordoutputLayer.b.set_value(numpy.asarray(b_dict[labels[i][1]], dtype = 'float32'), borrow = True)
                out = validate_model()
                log_likelihood.append(sum(out))
            valid_set_x1.set_value(numpy.empty((1), 'float32'))
            valid_set_x2.set_value(numpy.empty((1), 'float32'))
            valid_set_y_class.set_value(numpy.empty((1), 'int32'))
            valid_set_y_word.set_value(numpy.empty((1), 'int32'))
            progress += 1
            if progress%1000==0:
                end_time_valid_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)
        
        end_time_valid_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)
        end_time_valid_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)            
        entropy = (-numpy.sum(log_likelihood)/valid_frames_showed)
        print entropy, numpy.sum(log_likelihood)
        
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
            test_set_y_class.set_value(numpy.asarray([labels[i][1]], dtype = 'int32'), borrow = True)
            test_set_y_word.set_value(numpy.asarray([labels[i][0]], dtype = 'int32'), borrow = True)
            classifier.WordoutputLayer.W.set_value(numpy.asarray(w_dict[labels[i][1]], dtype = 'float32'), borrow = True)
            classifier.WordoutputLayer.b.set_value(numpy.asarray(b_dict[labels[i][1]], dtype = 'float32'), borrow = True)
            out = test_model()
            log_likelihood.append(sum(out))
        progress += 1
        if progress%1000==0:
           end_time_test_progress = time.time()
           print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, test_frames_showed, end_time_test_progress - start_test_time)
    end_time_test_progress = time.time()
    print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                    %(progress, test_frames_showed, end_time_test_progress - start_test_time)            
    #save_posteriors(log_likelihood, likelihoods, weight_path+file_name2)
    print numpy.sum(log_likelihood)
    likelihood_sum = (-numpy.sum(log_likelihood)/test_frames_showed)
    print 'entropy:', likelihood_sum

train_mlpclasses()
