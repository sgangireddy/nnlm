'''
Created on 14 Feb 2013

@author: s1264845
'''

import numpy
from vocab_create import Vocabulary

class FetProvider(object):
    
    def __init__(self, path_name, vocab, vocab_size): #constructor
        
        self.path_name = path_name
        self.vocab = vocab
        self.vocab_size = vocab_size
    
    #def __iter__(self):
     #   return self
    
    #def reset(self):
        #super(FetProvider, self).reset()
     #   self.utt_skipped = 0
    
    #def next(self):
    def bin_rep(self):    
        f = open(self.path_name, 'r')
        count = 0

        target = []
        features = []
                
        for line in f:
            line = line.split()
            for iter in xrange(1):
                line.insert(0, '<s>')
            for iter in xrange(1):
                line.append('</s>')

            for iter in xrange(len(line) - 1):
                bi_gram = line[iter : iter + 2]
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
        print count
        #features_final = numpy.zeros([count, self.vocab_size])
        
        #for i, word_index in zip(xrange(count), features):
         #   features_final[i][word_index - 1] = 1
        return features, numpy.array(target) - 1


#path_name = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/python-learning/account/train'
#
#a = Vocabulary(path_name)
#a.vocab_create()
#print a.vocab_size, len(a.vocab)
#
#path_name = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/python-learning/account/test'
#
#b = FetProvider(path_name, a.vocab, a.vocab_size)
#
#features_ip, target = b.bin_rep()
#
#print features_ip.shape, target.shape 






