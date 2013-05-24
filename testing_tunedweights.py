'''
Created on Apr 2, 2013

@author: sgangireddy
'''
import theano
import theano.tensor as T
import numpy, time
from vocab_create import Vocabulary
from mlp_new_uni import MLP
import h5py, getopt, sys
from data_provider_modified import DataProvider
#from data_provider_new import DataProvider
from mlp_save import save_posteriors
from vocab_hash import Vocabularyhash

def testing_tunedweights(path, feature_dimension, context, hidden_size, weight_path, file_name1, model_number):
    
    x1 = T.fvector('x1')
    x2 = T.fvector('x2')
    y = T.ivector('y')
    
    #voc_list = Vocabulary(path + 'train_modified1')
    #voc_list.vocab_create()
    #vocab = voc_list.vocab
    #vocab_size = voc_list.vocab_size
    #short_list = voc_list.short_list
    #short_list_size = voc_list.short_list_size

    voc_list = Vocabularyhash('/exports/work/inf_hcrc_cstr_udialogue/siva/data_normalization/vocab/wlist5c.nvp')
    voc_list.hash_create()
    vocab = voc_list.voc_hash
    vocab_size = voc_list.vocab_size

    
    dataprovider_test = DataProvider(path + 'test_modified1_1m', vocab, vocab_size)
    #dataprovider_test = DataProvider(path + 'test', vocab, vocab_size , short_list)
    
    test_set_x1 = theano.shared(numpy.empty((1), dtype='float32'))
    test_set_x2 = theano.shared(numpy.empty((1), dtype='float32'))
    test_set_y = theano.shared(numpy.empty((1), dtype = 'int32'))
    
    rng = numpy.random.RandomState() 
   
    classifier = MLP(rng = rng, input1 = x1, input2 = x2,  n_in = vocab_size, fea_dim = int(feature_dimension), context_size = int(context), n_hidden = int(hidden_size) , n_out = vocab_size)
    
    log_likelihood = classifier.sum(y)
    likelihood = classifier.likelihood(y)
    
    #test_model
    test_model = theano.function(inputs = [], outputs = [log_likelihood, likelihood],  \
                                 givens = {x1: test_set_x1,
                                           x2: test_set_x2,
                                           y: test_set_y})
    
    classifier_name = 'MLP' + str(model_number)
    
    f = h5py.File(weight_path+file_name1, "r")
    for i in xrange(0, classifier.no_of_layers, 2):
        path_modified = '/' + classifier_name + '/layer' + str(i/2)
        if i == 4:
	   classifier.params[i].set_value(numpy.asarray(f[path_modified + "/W"].value, dtype = 'float32'), borrow = True)  
        else:
           classifier.params[i].set_value(numpy.asarray(f[path_modified + "/W"].value, dtype = 'float32'), borrow = True)
           classifier.params[i + 1].set_value(numpy.asarray(f[path_modified + "/b"].value, dtype = 'float32'), borrow = True)
    f.close()
    
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
    #save_posteriors(log_likelihood, likelihoods, path_name+exp_name2)
 
    print numpy.sum(log_likelihood)
    entropy = (-numpy.sum(log_likelihood)/test_frames_showed)
    print 'entropy', entropy

path= '/exports/work/inf_hcrc_cstr_udialogue/siva/data/'

opts, extraparams = getopt.getopt(sys.argv[1:], "f:c:h:p:m:n:", ["--feature", "--context", "--hidden", "--path", "--file_name1", "--model_number"])
for o,p in opts:
  if o in ['-f','--feature']:
     feature_dimension = p
  elif o in ['-c', '--context']:
     context = p
  elif o in ['-h', '--hidden']:
     hidden_size = p
  elif o in ['-p', '--path']:
     weight_path = p
  elif o in ['-m', '--file_name1']:
     file_name1 = p
  elif o in ['-n', '--model_number']:
     model_number = p
 
testing_tunedweights(path, feature_dimension, context, hidden_size, weight_path, file_name1, model_number) 
