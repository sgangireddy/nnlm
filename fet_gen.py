'''
Created on 6 Feb 2013

@author: s1264845
'''
#from __future__ import print_function
import numpy
import time

def vocab_gen(path_name):
    vocab = {}
    f = open (path_name + 'test', 'r')
   # f_w = open ('./vocab', 'w')
    for line in f:
        line = line.split()
        for unigram in line:
            if vocab == {}:
                vocab[unigram] = 1
                vocab['</s>'] = 1
                vocab['<s>'] = 1
            elif vocab.has_key(unigram):
                vocab[unigram] = vocab[unigram] + 1
            else:
                vocab[unigram] = 1
    f.close()
    i =1
    for unigram in sorted(vocab.keys()):
        vocab[unigram] = i
        i = i + 1
    return vocab, i-1

def fet_gen(path_name, vocab, vocab_size):
    t0 = time.clock()
    f = open (path_name + 'test', 'r')
    target_final= numpy.zeros([1,vocab_size])
    features_final= numpy.zeros([1,vocab_size])
    for line in f:
        line = line.split()
        for iter in xrange(1):
            line.insert(0, '<s>')
        for iter in xrange(1):
            line.append('</s>')
            #print line
        for iter in xrange(len(line) - 1):
            bi_gram = line[iter : iter + 2]
            i = 0
            features = numpy.zeros([1, vocab_size])
            target  = numpy.zeros([1, vocab_size])
            for word in bi_gram:
                index = vocab[word]
                if i == 1:
                    target[0][index - 1] = 1
                    target_final = numpy.vstack((target_final, target))
                else:
                    features[0][index-1] = 1
                    features_final = numpy.vstack((features_final, features))
                    i = i + 1
    print target_final.shape, features_final.shape
    print time.clock() - t0
   # print target
path_name = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/python-learning/'

vocab, vocab_size = vocab_gen(path_name)
fet_gen(path_name, vocab, vocab_size)

