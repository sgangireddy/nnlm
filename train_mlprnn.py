import theano
import theano.tensor as T
import time, numpy
from vocab_create import Vocabulary
from data_provider_modified import DataProvider
import os, sys, getopt
from learn_rates import LearningRateNewBob, LearningRateList
from learn_rates import LearningRate
from logistic_regression import LogisticRegression
from mlprnn import MLP_RNN
from cache import TNetsCacheSimple, TNetsCacheLastElem
from numpy.core.numeric import dtype
from utils import GlobalCfg
import h5py
#from mlp_save import save_mlp, save_posteriors, save_learningrate
#from vocab_hash import Vocabularyhash
      
def train_mlprnn(weight_path = sys.argv[1], file_name1 = sys.argv[2], L1_reg = 0.0, L2_reg = 0.0000, path_name = '/exports/work/inf_hcrc_cstr_udialogue/siva/data/'): 

    voc_list = Vocabulary(path_name + 'train')
    voc_list.vocab_create()
    vocab = voc_list.vocab
    vocab_size = voc_list.vocab_size
    
    dataprovider_train = DataProvider(path_name + 'train', vocab, vocab_size)
    dataprovider_valid = DataProvider(path_name + 'valid', vocab, vocab_size )
    dataprovider_test = DataProvider(path_name + 'test', vocab, vocab_size )
    
    print '..building the model'

    #symbolic variables for input, target vector and batch index
    index = T.lscalar('index')
    x1 = T.fvector('x1')
    x2 = T.fvector('x2')
    x3 = T.fvector('x3')
    ht1 = T.fvector('ht1')
    y = T.ivector('y')
    learning_rate = T.fscalar('learning_rate') 

    #theano shared variables for train, valid and test
    train_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_x3 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    train_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    valid_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_x3 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    valid_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    test_set_x1 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_x2 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_x3 = theano.shared(numpy.empty((1), dtype='float32'), allow_downcast = True)
    test_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    rng = numpy.random.RandomState() 
   
    classifier = MLP_RNN(rng = rng, input1 = x1, input2 = x2,  input3 = x3, initial_hidden = ht1, n_in = vocab_size, fea_dim = int(sys.argv[3]), context_size = 2, n_hidden = int(sys.argv[4]) , n_out = vocab_size)
    
    hidden_state = theano.shared(numpy.empty((int(sys.argv[4]), ), dtype = 'float32')) 
    
    cost = classifier.cost(y)
    
    #constructor for learning rate class
    learnrate_schedular = LearningRateNewBob(start_rate = 0.05, scale_by=.5, max_epochs=9999,\
                                    min_derror_ramp_start=.01, min_derror_stop=.01, init_error=100.)

    log_likelihood = classifier.sum(y)
    likelihood = classifier.likelihood(y)
    
    #test_model
    test_model = theano.function(inputs = [], outputs = [log_likelihood, likelihood],  \
                                 givens = {x1: test_set_x1,
                                           x2: test_set_x2,
                                           x3: test_set_x3,
                                           ht1: hidden_state,
                                           y: test_set_y})
    #validation_model
    validate_model = theano.function(inputs = [], outputs = [log_likelihood], \
                                     givens = {x1: valid_set_x1,
                                               x2: valid_set_x2,
                                               x3: valid_set_x3,
                                               ht1: hidden_state,        
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
    train_model = theano.function(inputs = [learning_rate], outputs = [cost, classifier.RNNhiddenlayer.output], updates = updates, \
                                 givens = {x1: train_set_x1,
                                           x2: train_set_x2,
                                           x3: train_set_x3,
                                           ht1: hidden_state,
                                           y: train_set_y})
    f = h5py.File(weight_path+file_name1, "r")
    for i in xrange(0, classifier.no_of_layers, 2):
        path_modified = '/' + 'MLP'+ str(2) + '/layer' + str(i/2)
        if i == 4:
           classifier.MLPparams[i].set_value(numpy.asarray(f[path_modified + "/W"].value, dtype = 'float32'), borrow = True)
        else:
           classifier.MLPparams[i].set_value(numpy.asarray(f[path_modified + "/W"].value, dtype = 'float32'), borrow = True)
           classifier.MLPparams[i + 1].set_value(numpy.asarray(f[path_modified + "/b"].value, dtype = 'float32'), borrow = True)
    f.close()

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
                temp_features3 = numpy.zeros(vocab_size, dtype = 'float32')
                temp_features1[temp[0]] = 1
                temp_features2[temp[1]] = 1
                temp_features3[temp[1]] = 1
                train_set_x1.set_value(numpy.asarray(temp_features1, dtype = 'float32'), borrow = True)
                train_set_x2.set_value(numpy.asarray(temp_features2, dtype = 'float32'), borrow = True)
                train_set_x3.set_value(numpy.asarray(temp_features2, dtype = 'float32'), borrow = True)
                train_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
                out = train_model(numpy.array(learnrate_schedular.get_rate(), dtype = 'float32'))
                hidden_state.set_value(numpy.asarray(out[1], dtype = 'float32'), borrow = True)
                
            progress += 1
            if progress%10000==0:
                end_time_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
            train_set_x1.set_value(numpy.empty((1), dtype = 'float32'))
            train_set_x2.set_value(numpy.empty((1), dtype = 'float32'))
            train_set_x3.set_value(numpy.empty((1), dtype = 'float32'))
            train_set_y.set_value(numpy.empty((1), dtype = 'int32'))
        
        end_time_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
                          
                          
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
                temp_features3 = numpy.zeros(vocab_size, dtype = 'float32')
                temp_features1[temp[0]] = 1
                temp_features2[temp[1]] = 1
                temp_features3[temp[1]] = 1
                valid_set_x1.set_value(numpy.asarray(temp_features1, dtype = 'float32'), borrow = True)
                valid_set_x2.set_value(numpy.asarray(temp_features2, dtype = 'float32'), borrow = True)
                valid_set_x3.set_value(numpy.asarray(temp_features3, dtype = 'float32'), borrow = True)
                valid_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
                out = validate_model()
                #error_rate = out[0]
                likelihoods = out[0] 
                #valid_losses.append(error_rate)
                log_likelihood.append(likelihoods)
            valid_set_x1.set_value(numpy.empty((1), 'float32'))
            valid_set_y.set_value(numpy.empty((1), 'int32'))

            progress += 1
            if progress%1000==0:
                end_time_valid_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)
        
        end_time_valid_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)            
        #this_validation_loss = numpy.mean(valid_losses)
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
            temp_features3 = numpy.zeros(vocab_size, dtype = 'float32')
            temp_features1[temp[0]] = 1
            temp_features2[temp[1]] = 1
            temp_features3[temp[1]] = 1
            test_set_x1.set_value(numpy.asarray(temp_features1, dtype = 'float32'), borrow = True)
            test_set_x2.set_value(numpy.asarray(temp_features2, dtype = 'float32'), borrow = True)
            test_set_x3.set_value(numpy.asarray(temp_features3, dtype = 'float32'), borrow = True)
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
    print numpy.sum(log_likelihood)
    
train_mlprnn()
