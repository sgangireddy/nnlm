'''
Created on May 8, 2013

@author: sgangireddy
'''
import numpy, math

class Vocabulary(object):
    
    def __init__(self, path_name, n_classes): #constructor
        
        self.path_name = path_name
        self.vocab = {}
        self.vocab_prob = {}
        self.vocab_size = 0
        self.count = 0
        self.n_classes = n_classes
    
    def vocab_create(self):
        f = open (self.path_name, 'r')
        for line in f:
            line = line.strip().split()
            for unigram in line:
                self.count = self.count + 1
                if self.vocab == {}:
                    self.vocab[unigram] = 1
                    self.vocab['<s>'] = 1
                    self.vocab['</s>'] = 1
                elif self.vocab.has_key(unigram):
                    self.vocab[unigram] = self.vocab[unigram] + 1
                else:
                    self.vocab[unigram] = 1
        f.close()
        for unigram in self.vocab.keys():
            self.vocab_prob[unigram] = float(self.vocab[unigram]) / self.count
            
        for unigram in self.vocab.keys():
            self.vocab[unigram] = self.vocab_size
            self.vocab_size = self.vocab_size + 1

            
    def class_label(self):
        self.prob_sum = 0
        self.classes = {}
        self.words = {}
        self.percent = 1 / float(self.n_classes)
        i, j = 0, 0
        for word in sorted(self.vocab_prob, key=self.vocab_prob.get, reverse=True):
            self.prob_sum = self.prob_sum + self.vocab_prob[word]
            self.words[word] = i
            i = i + 1           
            if self.prob_sum > (self.percent):
                self.classes[j] = self.words
                j = j + 1
                self.words = {}
                i = 0
                self.prob_sum = 0
        self.classes[j] = self.words
        
#path_name = '/Users/sgangireddy/Documents/workspace/data/train'
#a = Vocabulary(path_name, 20)
#a.vocab_create()
#a.class_label()
#print len(a.vocab), len(a.classes)

#temp1 = []
#temp2 = []
#no_class = 5
#percen = 1 / float(no_class) 
#sum = 0
#sum_temp = 0
#i = 1
#
#
#for w in sorted(a.vocab_prob, key=a.vocab_prob.get, reverse=True):
#    sum = sum + a.vocab_prob[w]
#    temp1.append(w)
#    if sum >= (percen * i):
#        print len(temp1)
#        i = i + 1
#        sum_temp = sum_temp + len(temp1)
#        temp2.append(temp1)
#        temp1 = []
#print sum_temp, i, temp2[1]