'''
Created on May 13, 2013

@author: sgangireddy
'''
import numpy
#from vocab_create import Vocabulary

class DataProvider(object):
    
    def __init__(self, path_name, vocab, vocab_size): #constructor
        
        self.path_name = path_name
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.i = -1
        self.ngram = 2
        
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
            self.words.append('</s>')
            
            for iterator in xrange(len(self.words) - (self.ngram -1)):
                self.ngrams.append(self.words[iterator : iterator + self.ngram])
                        
            for gram in self.ngrams:
                features =[]
                for j in xrange(self.ngram):
                    if j == self.ngram - 1:
                        labels.append(self.vocab[gram[j]])
                    else:
                        features.append(self.vocab[gram[j]])
                fea_vec.append(numpy.array(features).flatten())
        else:
            raise StopIteration    

        return (numpy.asarray(fea_vec, dtype = 'float32'), numpy.asarray(labels, dtype = 'int32'))