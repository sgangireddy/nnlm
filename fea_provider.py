'''
Created on 8 Feb 2013

@author: s1264845
'''

'''
generates the binary representations of words using 1 of N coding
'''
import numpy

class FetProvider(object):
    
    def __init__(self, path_name): #constructor
        
        self.path_name = path_name
        self.vocab = {}
        self.vocab_size = 1
        
    
    def vocab_create(self):
        f = open (self.path_name, 'r')
        for line in f:
            line = line.strip().split()
            for unigram in line:
                if self.vocab == {}:
                    self.vocab[unigram] = 1
                    self.vocab['<s>'] = 1
                    self.vocab['</s>'] = 1
                elif self.vocab.has_key(unigram):
                    self.vocab[unigram] = self.vocab[unigram] + 1
                else:
                    self.vocab[unigram] = 1
        self.vocab['<unk>'] =  1
        f.close()
        for unigram in sorted(self.vocab.keys()):
            self.vocab[unigram] = self.vocab_size
            self.vocab_size = self.vocab_size + 1
        self.vocab_size = self.vocab_size - 1
       # return self.vocab_size
    
    def bin_rep(self):
        
        self.vocab_create()
        
        f = open(self.path_name, 'r')
        
        features_final = numpy.zeros([1, self.vocab_size])
        #target  = numpy.zeros([self.vocab_size, ])
        target = []
                
        for line in f:
            line = line.split()
            for iter in xrange(1):
                line.insert(0, '<s>')
            for iter in xrange(1):
                line.append('</s>')

            for iter in xrange(len(line) - 1):
                bi_gram = line[iter : iter + 2]
                i = 0
                features = numpy.zeros([1, self.vocab_size])
                for word in bi_gram:
                    index = self.vocab[word]
                    if i == 1:
                        target.append(index)
                    else:
                        features[0][index-1] = 1
                        features_final = numpy.vstack((features_final, features))
                        i = i + 1
        return features_final[1:,:], numpy.array(target) - 1
       
    
#path_name = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/python-learning/valid'
#generate_features = FetProvider(path_name)
#features_final, target_final = generate_features.bin_rep()
#print features_final.dtype, target_final.dtype
#print target_final[0:10]


