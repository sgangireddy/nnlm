
import numpy, theano
import h5py


def save_mlp(classifier, path, classifier_name):

    f = h5py.File(path, "a")
    for i in xrange(0, classifier.no_of_layers, 2):
	path_modified = '/' + classifier_name + '/layer' + str(i/2)
	if i == 4:
            f[path_modified + "/W"] = classifier.params[i].get_value(borrow = True)
        else:
            f[path_modified + "/W"] = classifier.params[i].get_value(borrow = True)
            f[path_modified + "/b"] = classifier.params[i + 1].get_value(borrow = True)
    f.close()
	
    return None

def save_posteriors(log_posteriors, posteriors, path):
    
    f = h5py.File(path, "w")
    f['/log_posteriors'] = numpy.array(log_posteriors, dtype = 'float32')
    f['/posteriors'] = numpy.array(posteriors, dtype = 'float32')
    f.close()
    
    return None
def save_learningrate(learning_rate, path, classifier_name):

    f = h5py.File(path, "a")
    f['/' + classifier_name] = numpy.array([learning_rate], dtype = 'float32') 














#path = '/afs/inf.ed.ac.uk/user/s12/s1264845/scratch/s1264845/mlp/nnets/finetuning.hdf5'

#f = h5py.File(path, "a")	   
