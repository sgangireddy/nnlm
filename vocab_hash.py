import sys


class Vocabularyhash(object):

	def __init__ (self, path):
	   self.voc_hash = {}
	   self.path = path
	   self.vocab_size = 0


	def hash_create(self):
	    f = open (self.path, 'r')
	    for line in f:
	        tmp = line.strip().split()
                if len(tmp) == 1:
                   self.voc_hash[tmp[0].lower()] = 1
   	    self.voc_hash['<s>'] = 1
    	    self.voc_hash['</s>'] = 1
   	    self.voc_hash['<unk>'] = 1

            for unigram in self.voc_hash.keys():
		self.voc_hash[unigram] = self.vocab_size
		self.vocab_size = self.vocab_size + 1

#path = '/exports/work/inf_hcrc_cstr_udialogue/siva/data_normalization/vocab/wlist5c.nvp' 
#a= Vocabularyhash(path)
#a.hash_create()
#print a.vocab_size
