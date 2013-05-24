'''
Created on 19 Feb 2013

@author: s1264845
'''
import numpy
from Queue import Queue
from Queue import Empty as QueueEmptyException

class cache(object):
    
    def __init__(self, queue, input_index, target_index, vocab_size, batch_size, num_batches_per_bunch, num_bunches_queue):
        
        self.queue = queue
        self.input_index = input_index
        self.target_index = target_index
        self.vocab_size = vocab_size
        self.batch_size =  batch_size
        self.num_batches_per_bunch = num_batches_per_bunch
        self.num_bunches_queue = num_bunches_queue
    
    @staticmethod
    def make_queue(maxsize = 10):
        return Queue(maxsize = maxsize)
    
    @staticmethod
    def get_elem_from_queue(queue):
        while True:
            try:
                result = queue.get(True, 5)
                return result
            except QueueEmptyException:
                continue
    
    def get_bunch(self, bunch_index):
        
        features = numpy.zeros([(self.batch_size * self.num_batches_per_bunch), self.vocab_size])
        
        for iterator in xrange(bunch_index, bunch_index + self.num_bunches_queue):
        
            input_index = self.input_index[iterator * (self.batch_size * self.num_batches_per_bunch) : (iterator + 1) * self.batch_size * self.num_batches_per_bunch]
    
            for i, word_index in zip(xrange(self.batch_size * self.num_batches_per_bunch), input_index):
                features[i][word_index - 1] = 1
        
            labels = self.target_index[iterator * (self.batch_size * self.num_batches_per_bunch) : (iterator + 1) * self.batch_size * self.num_batches_per_bunch]
            
            self.queue.put((numpy.asarray(features, dtype = numpy.float64), numpy.asarray(labels, dtype = numpy.int32)), block = True) 

    
