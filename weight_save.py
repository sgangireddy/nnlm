'''
Created on Apr 2, 2013

@author: sgangireddy
'''
import h5py
import numpy

def weight_save(path):
    
    f = h5py.File(path + 'dset.h5', 'w')
    #f.create_group('/MyGroup')
    for i in xrange(100):
    	f['/MyGroup'+'/'+'dset' + str(i)] = numpy.zeros((6,4), dtype=numpy.int32)
    #dset = f['dset']
    #print dset[...]
    f.close()
    return None
    
path = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/s1264845/'
weight_save(path)
f = h5py.File(path + 'dset.h5', 'r')
#print f['/MyGroup'+'/'+'dset1'].value.ndim
#print f['/MyGroup'+'/'+'dset2'].value.ndim
for i in xrange(100):
    print f['/MyGroup'+'/'+'dset' + str(i)].value.ndim
#f.close()
