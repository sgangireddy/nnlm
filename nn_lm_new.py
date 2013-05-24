'''
Created on 21 Feb 2013

@author: s1264845
'''
import theano
import theano.tensor as T
import time, numpy
from vocab_create import Vocabulary
from data_provider import DataProvider
import os, sys
from learn_rates import LearningRateNewBob, LearningRateList
from learn_rates import LearningRate
from logistic_regression import LogisticRegression
from mlp import MLP
from mlp import HiddenLayer
from cache import TNetsCacheSimple, TNetsCacheLastElem
from numpy.core.numeric import dtype
from utils import GlobalCfg
from mlp_save import save_mlp, save_posteriors
import math        
def train_mlp(L1_reg = 0.0, L2_reg = 0.0000, num_batches_per_bunch = 512, batch_size = 1, num_bunches_queue = 5, offset = 0, path_name = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/s1264845/data/'):
    

    voc_list = Vocabulary(path_name + 'train')
    voc_list.vocab_create()
    vocab = voc_list.vocab
    vocab_size = voc_list.vocab_size
    
    voc_list_valid = Vocabulary(path_name + 'valid')
    voc_list_valid.vocab_create()
    count = voc_list_valid.count

    voc_list_test = Vocabulary(path_name + 'test')
    voc_list_test.vocab_create()
    no_test_tokens = voc_list_test.count
    print 'The number of sentenses in test set:', no_test_tokens
 
    #print 'number of words in valid data:', count 
    dataprovider_train = DataProvider(path_name + 'train', vocab, vocab_size )
    dataprovider_valid = DataProvider(path_name + 'valid', vocab, vocab_size )
    dataprovider_test = DataProvider(path_name + 'test', vocab, vocab_size )

    #learn_list = [0.1, 0.1, 0.1, 0.75, 0.5, 0.25, 0.125, 0.0625, 0]
    exp_name = 'fine_tuning.hdf5'
    posterior_path = 'log_likelihoods'
    print '..building the model'

    #symbolic variables for input, target vector and batch index
    index = T.lscalar('index')
    x = T.fmatrix('x')
    y = T.ivector('y')
    learning_rate = T.fscalar('learning_rate') 

    #theano shares variables for train, valid and test
    train_set_x = theano.shared(numpy.empty((1,1), dtype='float32'), allow_downcast = True)
    train_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    valid_set_x = theano.shared(numpy.empty((1,1), dtype='float32'), allow_downcast = True)
    valid_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    test_set_x = theano.shared(numpy.empty((1,1), dtype='float32'), allow_downcast = True)
    test_set_y = theano.shared(numpy.empty((1), dtype = 'int32'), allow_downcast = True)
    
    rng = numpy.random.RandomState(1234) 
   
    classifier = MLP(rng = rng, input = x, n_in = vocab_size, n_hidden1 = 30, n_hidden2 = 60 , n_out = vocab_size)
    #classifier = MLP(rng = rng, input = x, n_in = vocab_size, n_hidden = 60, n_out = vocab_size)

    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    
    #constructor for learning rate class
    learnrate_schedular = LearningRateNewBob(start_rate=0.001, scale_by=.5, max_epochs=9999,\
                                    min_derror_ramp_start=.1, min_derror_stop=.1, init_error=100.)

    #learnrate_schedular = LearningRateList(learn_list)

    frame_error = classifier.errors(y)
    likelihood = classifier.sum(y)

    #test model
    test_model = theano.function(inputs = [index], outputs = likelihood,  \
                                 givens = {x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                           y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #validation_model
    validate_model = theano.function(inputs = [index], outputs = [frame_error, likelihood], \
                                     givens = {x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                               y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    gradient_param = []
    #calculates the gradient of cost with respect to parameters 
    for param in classifier.params:
        gradient_param.append(T.cast(T.grad(cost, param), 'float32'))
        
    updates = []
    
    for param, gradient in zip(classifier.params, gradient_param):
        updates.append((param, param - learning_rate * gradient))
    
    #training_model
    train_model = theano.function(inputs = [index, theano.Param(learning_rate, default = 0.01)], outputs = cost, updates = updates, \
                                 givens = {x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                           y: train_set_y[index * batch_size:(index + 1) * batch_size]})
   

    #theano.printing.pydotprint(train_model, outfile = "pics/train.png", var_with_name_simple = True) 
    #path_save = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/s1264845/mlp/saved_weights/' 
    print '.....training'    
    best_valid_loss = numpy.inf    
    epoch = 1
    start_time = time.time()
    while(learnrate_schedular.get_rate() != 0):
	
	print 'learning_rate:', learnrate_schedular.get_rate()
        print 'epoch_number:', learnrate_schedular.epoch
        
        frames_showed, progress = 0, 0
        start_epoch_time = time.time()
        
        tqueue = TNetsCacheSimple.make_queue()
        cache = TNetsCacheSimple(tqueue, shuffle_frames = True, offset=0, \
                                 batch_size = batch_size, num_batches_per_bunch = num_batches_per_bunch) 
        cache.data_provider = dataprovider_train
        cache.start()
        
        train_cost = []
        while True:

            feats_lab_tuple = TNetsCacheSimple.get_elem_from_queue(tqueue)
            if isinstance(feats_lab_tuple, TNetsCacheLastElem):
                break
                
            features, labels = feats_lab_tuple                  
            train_set_x.set_value(features, borrow=True)
            train_set_y.set_value(numpy.asarray(labels.flatten(), dtype = 'int32'), borrow=True)
            
            frames_showed += features.shape[0]
            train_batches = features.shape[0]/batch_size
            #print train_batches
                #if there is any part left in utterance (smaller than a batch_size), take it into account at the end
            if(features.shape[0] % batch_size!=0 or features.shape[0] < batch_size): 
                train_batches += 1
            
            for i in xrange(train_batches):
                #train_cost.append(train_model(i, learnrate_schedular.get_rate()))
                train_model(i, learnrate_schedular.get_rate())               
            progress += 1
            if progress%10==0:
                end_time_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
        
        end_time_progress = time.time()
	print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
        train_set_x.set_value(numpy.empty((1,1), dtype = 'float32'))
        train_set_y.set_value(numpy.empty((1), dtype = 'int32'))
	classifier_name = 'MLP' + str(learnrate_schedular.epoch)
	
	save_mlp(classifier, GlobalCfg.get_working_dir()+exp_name , classifier_name)
                  
        print 'Validating...'
        valid_losses = []
	log_likelihood = []
        valid_frames_showed, progress = 0, 0
        start_valid_time = time.time() # it is also stop of training time
        #for feat_lab_tuple, path in HDFDatasetDataProviderUtt(devel_files_list, valid_dataset, randomize=False, max_utt=-10):  
        #    features, labels = feat_lab_tuple 
            
        tqueue = TNetsCacheSimple.make_queue()
        cache = TNetsCacheSimple(tqueue, offset = 0, num_batches_per_bunch = 16)

        #cache.deamon = True
        cache.data_provider = dataprovider_valid
        cache.start()
        
        #ex_num = 0
        
        while True:
                
            feats_lab_tuple = TNetsCacheSimple.get_elem_from_queue(tqueue)
            if isinstance(feats_lab_tuple, TNetsCacheLastElem):
                break
                    
            features, labels = feats_lab_tuple

            valid_frames_showed += features.shape[0]                
            valid_set_x.set_value(features, borrow=True)
            valid_set_y.set_value(numpy.asarray(labels.flatten(), 'int32'), borrow=True)
            
            valid_batches = features.shape[0] / batch_size
            #print valid_batches
            #if there is any part left in utterance (smaller than a batch_size), take it into account at the end
            if(features.shape[0] % batch_size!=0 or features.shape[0] < batch_size): 
                valid_batches += 1
          
            for i in xrange(valid_batches):
                #ex_num = ex_num + 1
                out = validate_model(i)
		error_rate = out[0]
		likelihoods = out[1] 
                valid_losses.append(error_rate)
		log_likelihood.append(likelihoods)
                #save_posteriors(likelihoods, GlobalCfg.get_working_dir() + posterior_path, str(ex_num), str(learnrate_schedular.epoch))
                
	    
            progress += 1
            if progress%10==0:
                end_time_valid_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)
        
        end_time_valid_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)            
        valid_set_x.set_value(numpy.empty((1,1), 'float32'))
        valid_set_y.set_value(numpy.empty((1), 'int32'))
            
            
        end_epoch_time = time.time()
	print 'time taken for this epoch in seconds: %f' %(end_epoch_time - start_epoch_time)
            
        this_validation_loss = numpy.mean(valid_losses)
	loglikelihood_sum = numpy.sum(log_likelihood)
	#ppl = math.exp(- loglikelihood_sum /count)
	#print 'ppl:', ppl
	print 'error_rate:', this_validation_loss
	print 'valid log likelihood:', loglikelihood_sum     
	#print 'mean log_probability', this_validation_loss 
        #learnrate_schedular.get_next_rate(this_validation_loss * 100.)
	    #learnrate_schedular.get_next_rate()
	    #print 'epoch_number:', learnrate_schedular.epoch
                
            # logger.info('Epoch %i (lr: %f) took %f min (SPEED [presentations/second] training %f, cv %f), cv error %f %%' % \
            #         (self.cfg.finetune_scheduler.epoch-1, self.cfg.finetune_scheduler.get_rate(), \
            #          ((end_epoch_time-start_epoch_time)/60.0), (frames_showed/(start_valid_time-start_epoch_time)), \
            #          (valid_frames_showed/(stop_valid_time-start_valid_time)), this_validation_loss*100.))

            #self.cfg.finetune_scheduler.get_next_rate(this_validation_loss*100.)
        if this_validation_loss < best_valid_loss:
	   learning_rate = learnrate_schedular.get_next_rate(this_validation_loss * 100.)
           best_valid_loss = this_validation_loss
           #best_epoch = learnrate_schedular.epoch-1
	else:
           #learnrate_schedular.epoch = learnrate_schedular.epoch + 1
	   learnrate_schedular.rate = 0.0
    
    end_time = time.time()
        
    #print 'Optimization complete with best validation score of %f %%' %  best_valid_loss * 100.
    print 'The fine tuning ran for %.2fm' %((end_time-start_time)/60.)

    print 'Testing...'
    log_likelihood_test = []
    test_frames_showed, progress = 0, 0
    start_test_time = time.time() # it is also stop of training time
        #for feat_lab_tuple, path in HDFDatasetDataProviderUtt(devel_files_list, valid_dataset, randomize=False, max_utt=-10):  
        #    features, labels = feat_lab_tuple 
            
    tqueue = TNetsCacheSimple.make_queue()
    cache = TNetsCacheSimple(tqueue, offset = 0, num_batches_per_bunch = 16)

    #cache.deamon = True
    cache.data_provider = dataprovider_test
    cache.start()
        
        #ex_num = 0
        
    while True:
                
	feats_lab_tuple = TNetsCacheSimple.get_elem_from_queue(tqueue)
        if isinstance(feats_lab_tuple, TNetsCacheLastElem):
           break
                    
        features, labels = feats_lab_tuple

        test_frames_showed += features.shape[0]                
        test_set_x.set_value(features, borrow=True)
        test_set_y.set_value(numpy.asarray(labels.flatten(), 'int32'), borrow=True)
            
        test_batches = features.shape[0] / batch_size
            #print valid_batches
            #if there is any part left in utterance (smaller than a batch_size), take it into account at the end
        if(features.shape[0] % batch_size!=0 or features.shape[0] < batch_size): 
           test_batches += 1
          
        for i in xrange(test_batches): 
            log_likelihood_test.append(test_model(i))
	    
        progress += 1
        if progress%10==0:
           end_time_test_progress = time.time()
           print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, test_frames_showed, end_time_test_progress - start_test_time)
        
    end_time_test_progress = time.time()
    print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                    %(progress, test_frames_showed, end_time_test_progress - start_test_time)            
    test_set_x.set_value(numpy.empty((1,1), 'float32'))
    test_set_y.set_value(numpy.empty((1), 'int32'))
 
    likelihood_sum = numpy.sum(log_likelihood_test)
    print 'likelihood_sum', likelihood_sum
    #test_ppl = math.exp(- likelihood_sum / no_test_tokens)
    #print 'test_ppl:', test_ppl	

train_mlp()
