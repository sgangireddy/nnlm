'''
Created on 10 Sep 2012

@author: s1136550
'''

import random, time, thread, sys, numpy

from conf_global import *
#from utils import normalise_matrix, generate_minibatch

from Queue import Empty as QueueEmptyException

TNETS_LOADING_MODE = None
if TNETS_MUTLITASKING_MODE==TNETS_THREADS:
    from threading import Thread as TNETS_LOADING_MODE
    from Queue import Queue
elif TNETS_MUTLITASKING_MODE==TNETS_PROCESSES:
    from multiprocessing import Process as TNETS_LOADING_MODE
    from multiprocessing import Queue
assert TNETS_LOADING_MODE!=None


#class TNetsQueue(QueueClass):
#    def __init__(self, maxsize=10):
#        super(TNetsQueue, self).__init__(maxsize=maxsize)
#
#    def get_bunch(self, producer):
#        result = None
#        while (producer.is_alive() is True) or (not self.empty()):
#            try:
#                result = self.get(True, 5)
#                return result
#            except QueueEmptyException:
#                print 'Queue empty!!!'
#                pass
#        print 'Leaving as ', producer.is_alive(), self.empty()
#        return result
       
class TNetsCacheLastElem(object):
    def __init__(self):
        pass

class TNetsCache(TNETS_LOADING_MODE):
    def __init__(self, queue, shuffle_frames = True, offset = 0, batch_size = 8, num_batches_per_bunch = 4098, max_num_bunches_in_queue = 10):
        super(TNetsCache, self).__init__()
        self.queue = queue
        self.batch_size = batch_size
        self.num_batches_per_bunch = num_batches_per_bunch
        self.max_num_bunches_in_queue = max_num_bunches_in_queue
        self.data_provider = None
        self.is_loading = True
        self.shuffle_frames = shuffle_frames
        self.offset = offset
        
    def run(self):
        pass
    
    def is_running(self):
        return self.is_loading
    
    def shuffle_inplace(self, mat_a, mat_b):
        assert len(mat_a) == len(mat_b)
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(mat_a)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(mat_b)

    def shuffle_with_copy(self, mat_a, mat_b):
        assert len(mat_a) == len(mat_b)
        shuffled_a = numpy.empty(mat_a.shape, dtype=mat_a.dtype)
        shuffled_b = numpy.empty(mat_b.shape, dtype=mat_b.dtype)
        permutation = numpy.random.permutation(len(mat_a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = mat_a[old_index]
            shuffled_b[new_index] = mat_b[old_index]
        return shuffled_a, shuffled_b
    
    @staticmethod
    def make_queue(maxsize = 5):
        return Queue(maxsize = maxsize)
    
    @staticmethod
    def get_elem_from_queue(queue):
        while True:
            try:
                result = queue.get(True, 5)
                return result
            except QueueEmptyException:
                continue
        
class TNetsCacheSimple(TNetsCache):

    def __init__(self, queue, shuffle_frames  = True, offset = 0, batch_size = 8, num_batches_per_bunch = 4098, max_num_bunches_in_queue = 10):
        super(TNetsCacheSimple, self).__init__(queue, shuffle_frames, offset, batch_size, num_batches_per_bunch, max_num_bunches_in_queue)

    def run(self):
        
        bunch_size = self.batch_size * self.num_batches_per_bunch
        #print bunch_size, self.batch_size, self.num_batches_per_bunch, self.offset
        frames_count = 0
        tmp_list_feats, tmp_list_labels = [], []
        
#        count = 0
        
        for feat_lab_tuple in self.data_provider:
            #count = count + 1
            features, labels = feat_lab_tuple 
            if labels is None or features is None:
                continue
            #print features.shape
            read_in = frames_count + features.shape[0]
            if read_in < bunch_size:
                frames_count += features.shape[0]
                #print 'offset:', self.offset
                tmp_list_feats.append(generate_minibatch(features, self.offset))
                tmp_list_labels.append(labels.flatten())
            else:
                to_comp = bunch_size-frames_count
                #print 'to_comp', to_comp
                #print 'offset:', self.offset
                if to_comp>0:
                    tmp_list_feats.append(generate_minibatch(features[0:to_comp,:], self.offset))
                    tmp_list_labels.append(labels.flatten()[0:to_comp])
                feats_bunch = numpy.concatenate(tmp_list_feats, axis=0)
                labels_bunch = numpy.concatenate(tmp_list_labels)
                if (self.shuffle_frames is True):
                    feats_bunch, labels_bunch = self.shuffle_with_copy(feats_bunch, labels_bunch)

                self.queue.put((feats_bunch, labels_bunch), block=True)
                
                #print "TNetsCacheSimple: Pushed " + str(feats_bunch.shape) + str(labels_bunch.shape) +\
                #", %i bunch. %i frames goes to the next batch. %i" \
                #        %(self.queue.qsize(), features.shape[0] - to_comp, to_comp)
                
                #TODO: TEMPORAIRLY ASSUMED THAT THE BUNCH IS SET TO BE LARGE ENOUGH TO STORE THE WHOLE REMAINING PART FROM PREV. UTT
                tmp_list_feats, tmp_list_labels = [], []
                tmp_list_feats.append(generate_minibatch(features[to_comp:,:], self.offset))
                tmp_list_labels.append(labels.flatten()[to_comp:])
                frames_count = features.shape[0] - to_comp
                
        # make the last bunch, drop the tail which does not fit into a full batch 
        # (in the worse scenario, #batch_size-1 random examples are dropped in the whole dataset)
        if frames_count > self.batch_size:
            up_to = frames_count/self.batch_size * self.batch_size
            feats_bunch = numpy.concatenate(tmp_list_feats, axis=0)
            labels_bunch = numpy.concatenate(tmp_list_labels)
            self.queue.put((feats_bunch[0:up_to,:], labels_bunch[0:up_to]), block=True, timeout=60)
            #print 'Last bunch contained %i frames (%i normally)'%(up_to, bunch_size)
        
        #print 'Setting is loading to false'
        self.queue.put(TNetsCacheLastElem(), block=True)
        #print 'count', count     
        #print 'Finished!'

class TNetsCacheSimpleUnsup(TNetsCache):

    def __init__(self, queue, shuffle_frames=False, batch_size=100, num_batches_per_bunch=256, max_num_bunches_in_queue=10):
        super(TNetsCacheSimpleUnsup, self).__init__(queue, shuffle_frames, batch_size, num_batches_per_bunch, max_num_bunches_in_queue)

    def run(self):
        
        bunch_size = self.batch_size*self.num_batches_per_bunch
        frames_count = 0
        tmp_list_feats = []
        
        for features, path in self.data_provider:
            
            if features is None:
                print 'Skipping ', path
                continue
                                        
            #print features.shape
            read_in = frames_count + features.shape[0]
            if read_in < bunch_size:
                frames_count += features.shape[0]
                tmp_list_feats.append(generate_minibatch(features, self.offset))

            else:
                to_comp = bunch_size-frames_count
                if to_comp>0:
                    tmp_list_feats.append(generate_minibatch(features[0:to_comp,:], self.offset))
                    feats_bunch = numpy.concatenate(tmp_list_feats, axis=0)
                if (self.shuffle_frames is True):
                    numpy.random.shuffle(feats_bunch)
                self.queue.put(feats_bunch, block=True)
                
                #print "TNetsCacheSimple: Pushed " + str(feats_bunch.shape) + str(labels_bunch.shape) +\
                #", %i bunch out of %i. %i frames goes to the next batch. %i" \
                #        %(self.queue.qsize(), self.queue.maxsize, features.shape[0] - to_comp, to_comp)
                
                #TODO: TEMPORAIRLY ASSUMED THAT THE BUNCH IS SET TO BE LARGE ENOUGH TO STORE THE WHOLE REMAINING PART FROM PREV. UTT
                tmp_list_feats = []
                tmp_list_feats.append(generate_minibatch(features[to_comp:,:], self.offset))
                frames_count = features.shape[0] - to_comp
        
        # make the last bunch, drop the tail which does not fit into a full batch 
        # (in the worse scenario, #batch_size-1 random examples are dropped in the whole dataset)
        if frames_count > self.batch_size:
            up_to = frames_count/self.batch_size * self.batch_size
            feats_bunch = numpy.concatenate(tmp_list_feats, axis=0)
            self.queue.put(feats_bunch[0:up_to,:], block=True)
            #print 'Last bunch contained %i frames (%i normally)'%(up_to, bunch_size)
        
        self.queue.put(TNetsCacheLastElem(), block=True)

def generate_minibatch(features, offset, add_paddings = True):
    """
    Function creates a batch to use with MLP/DNNs by adding to each row 
    an appropriate frame context defined by the offset and optional padding frames
    at the beginning and at the end of an array.
    
    :type features: numpy.ndarray
    :param features: 2D acoustic vectors num_frames x vector_size
    
    :type return: numpy.ndarray
    :param return:  the transformed features 2D array
    
    """
    
    if offset<0:
        return features;
    #print features  
    num_frames, vec_size = features.shape
    #print features.shape, vec_size.shape
    ctx_win = numpy.arange((offset*2+1) * vec_size)
    # print ctx_win.shape
    frames = numpy.arange(num_frames) * vec_size
    #print frames.shape
    indexes  = frames[:, numpy.newaxis] + ctx_win[numpy.newaxis, :]
    #print indexes
    
    inputs = features.flatten()
    #print inputs.shape
    if (add_paddings):
        padd_beg = numpy.tile(inputs[0:vec_size], offset)
        #print padd_beg.shape
        padd_end = numpy.tile(inputs[-vec_size:], offset)
        #print padd_end.shape
        inputs = numpy.concatenate((padd_beg, inputs, padd_end))
    #print inputs.shape
    
    return numpy.asarray(inputs[indexes], dtype=numpy.float32)
