'''
Created on Mar 26, 2013

@author: sgangireddy
'''
from cache import TNetsCacheSimple, TNetsCacheLastElem
import numpy
import time
from mlp_save import save_mlp

def training(dataprovider_train, dataprovider_valid, learnrate_schedular, classifier, train_model, validate_model, train_set_x, train_set_y, valid_set_x, valid_set_y, batch_size, num_batches_per_bunch, valid_words_count, valid_lines_count):
    
    exp_name = 'fine_tuning.hdf5' 
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
            frames_showed += features.shape[0]
            #train_batches = features.shape[0]/batch_size
            #print train_batches
                #if there is any part left in utterance (smaller than a batch_size), take it into account at the end
            #if(features.shape[0] % batch_size!=0 or features.shape[0] < batch_size): 
            #    train_batches += 1
            for i in xrange(features.shape[0]):
                temp_features = []
                for j in features[i]:
                    temp_features.append(classifier.projectionlayer.word_rep.get_value(borrow = True)[j])
                train_set_x.set_value(numpy.asarray(temp_features, dtype = 'float32').flatten(), borrow = True)
                train_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
                train_model(learnrate_schedular.get_rate())
                               
            progress += 1
            if progress%500==0:
                end_time_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
        
        end_time_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames), TIME: %f in seconds'\
                          %(progress, frames_showed,(end_time_progress-start_epoch_time))
        train_set_x.set_value(numpy.empty((1), dtype = 'float32'))
        train_set_y.set_value(numpy.empty((1), dtype = 'int32'))
	classifier_name = 'MLP' + str(learnrate_schedular.epoch)
	work_dir= '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/s1264845/mlp6/nnets'
	save_mlp(classifier, work_dir+exp_name , classifier_name)
        
        print 'Validating...'
        valid_losses = []
        log_likelihood = []
        valid_frames_showed, progress = 0, 0
        start_valid_time = time.time() # it is also stop of training time
                    
        tqueue = TNetsCacheSimple.make_queue()
        cache = TNetsCacheSimple(tqueue, offset = 0, num_batches_per_bunch = 512)

        cache.data_provider = dataprovider_valid
        cache.start()

        
        while True:
                
            feats_lab_tuple = TNetsCacheSimple.get_elem_from_queue(tqueue)
            if isinstance(feats_lab_tuple, TNetsCacheLastElem):
                break
                    
            features, labels = feats_lab_tuple

            valid_frames_showed += features.shape[0]                
            #valid_batches = features.shape[0] / batch_size
            #print valid_batches
            #if there is any part left in utterance (smaller than a batch_size), take it into account at the end
            #if(features.shape[0] % batch_size!=0 or features.shape[0] < batch_size): 
            #    valid_batches += 1
            
            for i in xrange(features.shape[0]):
                temp_features = []
                for j in features[i]:
                    temp_features.append(classifier.projectionlayer.word_rep.get_value(borrow = True)[j])
                valid_set_x.set_value(numpy.asarray(temp_features, dtype = 'float32').flatten(), borrow = True)
                valid_set_y.set_value(numpy.asarray([labels[i]], dtype = 'int32'), borrow = True)
                out = validate_model()
                error_rate = out[0]
                likelihoods = out[1] 
                valid_losses.append(error_rate)
                log_likelihood.append(likelihoods)
                #save_posteriors(likelihoods, GlobalCfg.get_working_dir() + posterior_path, str(ex_num), str(learnrate_schedular.epoch))
                
        
            progress += 1
            if progress%100==0:
                end_time_valid_progress = time.time()
                print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)
        
        end_time_valid_progress = time.time()
        print 'PROGRESS: Processed %i bunches (%i frames),  TIME: %f in seconds'\
                          %(progress, valid_frames_showed, end_time_valid_progress - start_valid_time)            
        valid_set_x.set_value(numpy.empty((1), 'float32'))
        valid_set_y.set_value(numpy.empty((1), 'int32'))
                
        end_epoch_time = time.time()
        print 'time taken for this epoch in seconds: %f' %(end_epoch_time - start_epoch_time)
            
        this_validation_loss = numpy.mean(valid_losses)
	entropy = (-numpy.sum(log_likelihood)/valid_frames_showed)
	print this_validation_loss, entropy, numpy.sum(log_likelihood)
        #loglikelihood_sum = numpy.sum(log_likelihood)
        #print 'error_rate:', this_validation_loss     
                        
        if entropy < best_valid_loss:
            #learning_rate = learnrate_schedular.get_next_rate(this_validation_loss * 100.)
	    learning_rate = learnrate_schedular.get_next_rate(entropy)
            best_valid_loss = entropy
            #best_epoch = learnrate_schedular.epoch-1
        else:
           #learnrate_schedular.epoch = learnrate_schedular.epoch + 1
           learnrate_schedular.rate = 0.0
    end_time = time.time()
    print 'The fine tuning ran for %.2fm' %((end_time-start_time)/60.)
