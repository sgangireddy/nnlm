'''
Created on Mar 26, 2013

@author: sgangireddy
'''
from cache import TNetsCacheSimple, TNetsCacheLastElem
import numpy
import time

def testing(dataprovider_test, classifier, test_model, test_set_x, test_set_y, test_words_count, test_lines_count, batch_size, num_batches_per_bunch):
    
    print 'Testing...'
    log_likelihood = []
    test_frames_showed, progress = 0, 0
    start_test_time = time.time() # it is also stop of training time
        #for feat_lab_tuple, path in HDFDatasetDataProviderUtt(devel_files_list, valid_dataset, randomize=False, max_utt=-10):  
        #    features, labels = feat_lab_tuple 
            
    tqueue = TNetsCacheSimple.make_queue()
    cache = TNetsCacheSimple(tqueue, offset = 0, num_batches_per_bunch = 512)

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
        #test_set_x.set_value(features, borrow=True)
        #test_set_y.set_value(numpy.asarray(labels.flatten(), 'int32'), borrow=True)
            
        test_batches = features.shape[0] / batch_size
            #print valid_batches
            #if there is any part left in utterance (smaller than a batch_size), take it into account at the end
        if(features.shape[0] % batch_size!=0 or features.shape[0] < batch_size): 
           test_batches += 1
          
        #for i in xrange(test_batches): 
        for i in xrange(features.shape[0]):
            temp_features = []
            for j in features[i]:
                temp_features.append(classifier.projectionlayer.word_rep.get_value(borrow = True)[j])
            test_set_x.set_value(numpy.asarray(temp_features, dtype = 'float32').flatten(), borrow = True)
            test_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
            log_likelihood.append(test_model())
        
        progress += 1
        if progress%100==0:
           end_time_test_progress = time.time()
           print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, test_frames_showed, end_time_test_progress - start_test_time)
        
    end_time_test_progress = time.time()
    print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                    %(progress, test_frames_showed, end_time_test_progress - start_test_time)            
    test_set_x.set_value(numpy.empty((1), 'float32'))
    test_set_y.set_value(numpy.empty((1), 'int32'))
 
    print numpy.sum(log_likelihood)
    likelihood_sum = (-numpy.sum(log_likelihood)/test_frames_showed)
    print 'likelihood_sum', likelihood_sum

