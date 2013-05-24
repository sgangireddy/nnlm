'''
Created on 14 Feb 2013

@author: s1264845
'''

class Vocabulary(object):
    
    def __init__(self, path_name): #constructor
        
        self.path_name = path_name
        self.vocab = {}
        self.short_list = {}
        self.short_list_freq = 0
        self.vocab_size = 0
        self.short_list_size = 0 
       	self.count = 0 
    
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
        #self.vocab['<unk>'] =  1
        f.close()
        for unigram in self.vocab.keys():
            if self.vocab[unigram] > 45:
                self.short_list_freq = self.short_list_freq + self.vocab[unigram]
                self.short_list[unigram] = self.short_list_size
                self.short_list_size = self.short_list_size + 1
        for unigram in self.vocab.keys():
            self.vocab[unigram] = self.vocab_size
            self.vocab_size = self.vocab_size + 1
        
#path_name = '/exports/work/inf_hcrc_cstr_udialogue/siva/data/train_modified1'
#a = Vocabulary(path_name)
#a.vocab_create()
#print a.vocab['<unk>'], a.vocab_size, len(a.vocab)
#print len(a.short_list)
#for unigram in a.short_list.keys():
#    print unigram, a.short_list[unigram]
#print a.short_list_freq, a.count
#for unigram in a.vocab:
#    print unigram
