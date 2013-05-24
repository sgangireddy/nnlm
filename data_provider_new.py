'''
Created on Mar 27, 2013

@author: sgangireddy
'''
import numpy
from vocab_create import Vocabulary
#from mlp import ProjectionLayer

class DataProvider(object):
    
    def __init__(self, path_name, vocab, vocab_size, short_list): #constructor
        
        self.path_name = path_name
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.short_list = short_list
        self.i = -1
        self.ngram = 3
        #rng = numpy.random.RandomState()
        #self.fea_con = ProjectionLayer(10000, 30, rng) 
        
        f = open(self.path_name, 'r')
        self.lines = f.readlines()
        f.close()
        
                    
    def __iter__(self):
        return self
    
    def reset(self):
        self.i = -1
        
    def next(self):    

        if self.i < len(self.lines) - 1:
            self.i = self.i + 1
            self.ngrams = []
            fea_vec = []
            labels = []
            self.words = self.lines[self.i].split()
            self.words.insert(0, '<s>')
            self.words.insert(0, '<s>')
            self.words.insert(0, '<s>')
            self.words.append('</s>')
            
            for iterator in xrange(len(self.words) - (self.ngram -1)):
                self.ngrams.append(self.words[iterator : iterator + self.ngram])
            
            temp = self.ngrams[0]
                      
            for gram in self.ngrams:
                features =[]
                if self.short_list.has_key(gram[self.ngram - 1]):
                    for j in xrange(self.ngram):                    
                        if j == self.ngram - 1:
                            labels.append(self.short_list[gram[j]])
                        else:                        
                            features.append(self.vocab[gram[j]])
                    for j in xrange(self.ngram):
                        features.append(self.vocab[temp[j]])
                    temp = gram
                    fea_vec.append(numpy.array(features).flatten())
                else:
                    temp = gram 
                    continue
        else:
            raise StopIteration    

        return (numpy.asarray(fea_vec, dtype = 'float32'), numpy.asarray(labels, dtype = 'int32'))
    
#path_name = '/Users/sgangireddy/Documents/workspace/data/valid'
###
#voc_list = Vocabulary(path_name)
#voc_list.vocab_create()
#vocab = voc_list.vocab
#vocab_size = voc_list.vocab_size
#short_list = voc_list.short_list
###
#a = DataProvider(path_name, vocab, vocab_size, short_list)
###
#for feat_lab_tuple in a:
#    features, labels = feat_lab_tuple
#    print features
#print 'one iteration completed'
#a.reset()
#for feat_lab_tuple in a:
#    features, labels = feat_lab_tuple
#print 'two iterations completed'

