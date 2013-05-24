'''
Created on May 18, 2013

@author: sgangireddy
'''
import numpy
from vocab_class import Vocabulary
#from mlp import ProjectionLayer

class DataProvider(object):
    
    def __init__(self, path_name, vocab, vocab_size, class_dict): #constructor
        
        self.path_name = path_name
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.i = -1
        self.ngram = 3
        self.class_dict = class_dict
        
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
            self.words.append('</s>')
            
            for iterator in xrange(len(self.words) - (self.ngram -1)):
                self.ngrams.append(self.words[iterator : iterator + self.ngram])
                        
            for gram in self.ngrams:
                features =[]
                labels_temp = []
                for j in xrange(self.ngram):
                    if j == self.ngram - 1:
                        #labels.append(self.vocab[gram[j]])
                        for i in xrange(len(self.class_dict)):
                            try:
                                labels_temp.append(self.class_dict[i][gram[j]])
                                labels_temp.append(i)
                                #labels_temp.append(len(self.class_dict[i]))
                                break
                            except KeyError:
                                pass
                    else:
                        features.append(self.vocab[gram[j]])
                fea_vec.append(numpy.array(features).flatten())
                labels.append(labels_temp)
        else:
            raise StopIteration    

        return (numpy.asarray(fea_vec, dtype = 'float32'), numpy.asarray(labels, dtype = 'int32'))
    
#path_name = '/Users/sgangireddy/Documents/workspace/data/train'
###
#voc_list = Vocabulary(path_name, 20)
#voc_list.vocab_create()
#voc_list.class_label()
#vocab = voc_list.vocab
#classes = voc_list.classes
#vocab_size = voc_list.vocab_size
###
#a = DataProvider(path_name, vocab, vocab_size, classes)
###
#for feat_lab_tuple in a:
#    features, labels = feat_lab_tuple
#    print labels
    #print a.ngrams
#print 'one iteration completed'
#a.reset()
#for feat_lab_tuple in a:
#    features, labels = feat_lab_tuple
#print 'two iterations completed'

#    
#    print features

