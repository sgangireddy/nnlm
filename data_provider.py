'''
Created on 25 Feb 2013

@author: s1264845
'''
import numpy
from vocab_create import Vocabulary
from cache import TNetsCacheSimple, TNetsCacheSimpleUnsup, TNetsCacheLastElem, TNetsCache

class DataProvider(object):
    
    def __init__(self, path_name, vocab, vocab_size): #constructor
        
        self.path_name = path_name
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.i = -1
        
        f = open(self.path_name, 'r')
        count = 0
        line_count = 0

        self.target_final = []
        self.features_final = []
        
        target = []
        features = []
                
        for line in f:
            line_count = line_count + 1
            line = line.split()
            for iterator in xrange(1):
                line.insert(0, '<s>')
            for iterator in xrange(1):
                line.append('</s>')

            for iterator in xrange(len(line) - 1):
                bi_gram = line[iterator : iterator + 2]
                count = count + 1
                i = 0
                for word in bi_gram:
                    if self.vocab.has_key(word):
                        index = self.vocab[word]
                    else:
                        index = self.vocab['<unk>']
                    
                    if i == 1:
                        target.append(index)
                    else:
                        features.append(index)
                        i = i + 1
            if line_count % 100 == 0:
                self.target_final.append(target)
                self.features_final.append(features)
                target = []
                features = []
        #print line_count
        self.target_final.append(target)
        self.features_final.append(features)    
            
    def __iter__(self):
        return self
        
    def next(self):    

        if self.i < len(self.features_final) - 1:
            self.i = self.i + 1
            features = numpy.zeros([len(self.features_final[self.i]), self.vocab_size], dtype = 'float32')
               
            input_index = self.features_final[self.i]
            labels = numpy.array(self.target_final[self.i], dtype = 'float32') - 1  
            
            for j, word_index in zip(xrange(len(self.features_final[self.i])), input_index):
                features[j][word_index] = 1
        else:
            raise StopIteration    

        return (numpy.asarray(features, dtype = 'float32'), numpy.asarray(labels, dtype = 'int32'))


#path_name = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/python-learning/'    
#voc_list = Vocabulary(path_name + 'train')
#voc_list.vocab_create()
#vocab = voc_list.vocab
#vocab_size = voc_list.vocab_size
#
#a = DataProvider(path_name + 'valid', vocab, vocab_size )
#
#batch_size = 100
#num_batches_per_bunch  = 50
#tqueue = TNetsCacheSimple.make_queue()
#cache = TNetsCacheSimple(tqueue, shuffle_frames = False)
#cache.data_provider = a
#cache.start()
#
#while True:
#        feats_lab_tuple = TNetsCacheSimple.get_elem_from_queue(tqueue)
#        features, labels = feats_lab_tuple
#        #print features.shape, labels.shape 
#        if isinstance(feats_lab_tuple, TNetsCacheLastElem):
#            break

