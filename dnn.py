'''
Created on 18 May 2012

@author: s1136550
'''

import numpy, theano
from tnets.dnnet.dbn import DBN
from tnets.dnnet.rbm import RBM
from tnets.io.dataset import Dataset

# these I/O DBNs functions are in a very draft stadium
def save_dbn(dbn, hd5_path, dbn_name, pretrained_only=True):
    
    import h5py
    
    def save_rbm(rbm, path, hd5_handler):
        
        hd5_handler[path+"/W"] = rbm.params[0].get_value(borrow=True)
        hd5_handler[path+"/hbias"] = rbm.params[1].get_value(borrow=True)
        hd5_handler[path+"/vbias"] = rbm.params[2].get_value(borrow=True)

        if(rbm.type == RBM.SRBM):
            hd5_handler[path+"/L"] = rbm.params[3].get_value(borrow=True)

        return None
    
    def save_layer(layer, path, hd5_handler):
        
        hd5_handler[path+"/W"] = layer.params[0].get_value(borrow=True)
        hd5_handler[path+"/b"] = layer.params[1].get_value(borrow=True)
        
        return None
            
    f = h5py.File(hd5_path, "a")
    path_dbn = '/'+dbn_name
    itr=0
    for rbm in dbn.pretrain_layers:
        path_rbm = path_dbn+'/rbm/l'+str(itr)
        save_rbm(rbm, path_rbm, f)
        itr+=1
    
    path_layer = path_dbn+'/linreg/'
    save_layer(dbn.logLayer,  path_layer, f)
       
    f.close()
    
    return None

def load_dbn_params(path_hd5, dbn_name):
    
    import h5py
    
    def load_rbm_params(path, hd5_handler):
        
        path_rbm = path+'/rbm/'
        rbms_group = list(hd5_handler[path_rbm])
        Ws, hbiases, vbiases = [], [], []
        for rbm_layer in rbms_group:
            W = numpy.array(hd5_handler[path_rbm+rbm_layer+'/W'], dtype=theano.config.floatX)
            hbias = numpy.array(hd5_handler[path_rbm+rbm_layer+'/hbias'], dtype=theano.config.floatX)
            vbias = numpy.array(hd5_handler[path_rbm+rbm_layer+'/vbias'], dtype=theano.config.floatX)
            Ws.append(W)
            hbiases.append(hbias)
            vbiases.append(vbias)
        
        return Ws, hbiases, vbiases
    
    def load_layer(path, hd5_handler):
        W = numpy.array(hd5_handler[path+'/W'], dtype=theano.config.floatX)
        bias = numpy.array(hd5_handler[path+'/b'], dtype=theano.config.floatX)
        
        return W, bias
    
    #load data
    f = h5py.File(path_hd5, "r")  
    path = '/'+dbn_name  
    Ws, hbiases, vbiases = load_rbm_params(path, f)
    W_linreg, b_linreg = load_layer(path+'/linreg', f)
    f.close()
    
    return Ws, hbiases, vbiases, W_linreg, b_linreg

def load_dbn(path_hd5, dbn_name, load_up_to_layer=-1, cfg=None, class_counts=None):
    
    Ws, hbiases, vbiases, W_linreg, b_linreg = load_dbn_params(path_hd5, dbn_name)
            
    #extract DBN structure
    n_in = Ws[0].shape[0]
    hid_layers_sizes = []
    for hbias in hbiases:
        hid_layers_sizes.append(hbias.shape[0])
    n_out = W_linreg.shape[1]
    
    up_to_idx=len(hbiases)
    if load_up_to_layer>0 :
        up_to_idx = load_up_to_layer
    
    # if the hidden layers have different dimensions and not the entire DBN structure is loaded, 
    # logistic regression layer (previously stored) may have wrong first dimension -> set weights to None 
    # so it will be created again from scratch with an appropriate first dimension. For pretrained only
    # structure it does not make any difference as the log reg layer weights are zeroed anyway.
    if (up_to_idx < len(hbiases) and (W_linreg.shape[0]!= Ws[up_to_idx-1].shape[1])):
        W_linreg, b_linreg = None, None
          
    #some pretrained layers could be shared and reused if we want to change some deeper part
    # e.g. one of logistic layer dimensions, add some extra random layers at the top for pretraining
    if cfg!=None:
        #change logistic layer
        if (cfg.output_size!=n_out): 
            print 'Resetting softmax size to', cfg.output_size
            n_out = cfg.output_size
            W_linreg, b_linreg = None, None #force to create new logistic layer
        
        #add some random layers at the top, generate new loglayer
        if (len(cfg.hidden_layers_sizes)>len(hid_layers_sizes)): 
            print 'Adding random layers at the top'
            for i in xrange(len(hid_layers_sizes), len(cfg.hidden_layers_sizes)):
                #print i, cfg.hidden_layers_sizes[i]
                hid_layers_sizes.append(cfg.hidden_layers_sizes[i])
                Ws.append(None)
                hbiases.append(None)
                vbiases.append(None)
                W_linreg, b_linreg = None, None #force to create the new logistic layer
            up_to_idx = len(hbiases)
        
        if cfg.finetune_top_N_layers>0: #we want learn softmax once more
            print 'Resetting softmax size as finetune_top_N_layers>0'
            W_linreg, b_linreg = None, None
            
        if cfg.reset_softmax is True:
            print 'Resetting softmax as reset_softmax=True'
            W_linreg, b_linreg = None, None
        
    return DBN(n_ins=n_in, hidden_layers_sizes=hid_layers_sizes[0:up_to_idx], n_outs=n_out,\
               W_list=Ws[0:up_to_idx], hbias_list=hbiases[0:up_to_idx], vbias_list=vbiases[0:up_to_idx],\
               W_lr_values=W_linreg, b_lr_values=b_linreg,\
               numpy_rng = numpy.random.RandomState(), cfg=cfg, class_counts=class_counts)

def load_dbn_new(path_hd5, dbn_name, load_up_to_layer=-1, cfg=None):
    
    Ws, hbiases, vbiases, W_linreg, b_linreg = load_dbn_params(path_hd5, dbn_name)
    
    if (load_up_to_layer > 0):
        Ws = Ws[0:load_up_to_layer]
        hbiases = hbiases[0:load_up_to_layer]
        vbiases = vbiases[0:load_up_to_layer]
    
    #extract DBN structure
    n_in = Ws[0].shape[0]
    hid_layers_sizes = []
    for hbias in hbiases:
        hid_layers_sizes.append(hbias.shape[0])
    n_out = W_linreg.shape[1]
    
    up_to_idx=len(hbiases)
    if load_up_to_layer>0 :
        up_to_idx = load_up_to_layer
        W_linreg, b_linreg = None, None
             
    #some pretrained layers could be shared and reused if we want to change some deeper part
    # e.g. one of logistic layer dimensions, add some extra random layers at the top for pretraining
    if cfg!=None:
        #change logistic layer
        if (cfg.output_size!=n_out): 
            n_out = cfg.output_size
            W_linreg, b_linreg = None, None #force to create new logistic layer
        
        #add some random layers at the top, generate new loglayer
        if (len(cfg.hidden_layers_sizes)>len(hid_layers_sizes)): 
            for i in xrange(len(hid_layers_sizes), len(cfg.hidden_layers_sizes)):
                #print i, cfg.hidden_layers_sizes[i]
                hid_layers_sizes.append(cfg.hidden_layers_sizes[i])
                Ws.append(None)
                hbiases.append(None)
                vbiases.append(None)
            W_linreg, b_linreg = None, None #force to create the new logistic layer
            up_to_idx = len(hbiases)
        
        if cfg.finetune_top_N_layers>0: #we want learn the softmax once more
            W_linreg, b_linreg = None, None
           
    return DBN(n_ins=n_in, hidden_layers_sizes=hid_layers_sizes[0:up_to_idx], n_outs=n_out,\
               W_list=Ws[0:up_to_idx], hbias_list=hbiases[0:up_to_idx], vbias_list=vbiases[0:up_to_idx],\
               W_lr_values=W_linreg, b_lr_values=b_linreg,\
               numpy_rng = numpy.random.RandomState(), cfg=cfg)

def save_to_mat(qn_weight_matfile, dnn):
    """These mat-weights could be read by QuickNet
    """
    
    import scipy.io
    
    qn_dict = {}
    for i in xrange(0,dnn.n_layers):
        qn_dict['weights%i%i'%(i+1, i+2)] = numpy.transpose(dnn.sigmoid_layers[i].W.get_value(borrow=True))
        qn_dict['bias%i'%(i+2)] = numpy.reshape(dnn.sigmoid_layers[i].b.get_value(borrow=True), (1,-1))
                                                    
    qn_dict['weights%i%i'%(i+2,i+3)] = numpy.transpose(dnn.logLayer.W.get_value(borrow=True)) 
    qn_dict['bias%i'%(i+3)] = numpy.reshape(dnn.logLayer.b.get_value(borrow=True), (1,-1))
    
    scipy.io.savemat(qn_weight_matfile, qn_dict, format='4') # 4th version is important for QN
    
    return None

def load_from_mat(qn_weight_matfile):
    """Load QuickNet weights
    """
    
    import scipy.io
    
    qnset = scipy.io.loadmat(qn_weight_matfile)
    layers = len(qnset)/2
    hbiases, Ws = [], []
    for i in xrange(0,layers):
        Ws.append(numpy.transpose(qnset['weights%i%i'%(i+1, i+2)]))
        hbiases.append(numpy.asarray(qnset['bias%i'%(i+2)]).flatten())
       
    #extract DBN structures
    n_in = Ws[0].shape[0]
    hid_layers_sizes = []
    #the last one layer is for regression
    for hbias in hbiases[:-1]:
        hid_layers_sizes.append(hbias.shape[0])
    n_out = Ws[-1].shape[1]
        
    return DBN(n_ins=n_in, hidden_layers_sizes=hid_layers_sizes, n_outs=n_out,\
               W_list=Ws[0:-1], hbias_list=hbiases[0:-1], vbias_list=None,\
               W_lr_values=Ws[-1], b_lr_values=hbiases[-1],\
               numpy_rng = numpy.random.RandomState())

def export_weights_to_kaldi(nnet_weight_file, dnn):
    
    f = open(nnet_weight_file, 'w')
    
    #hidden layers
    for i in xrange(dnn.n_layers): 
        
        W = dnn.sigmoid_layers[i].W.get_value()
        b = dnn.sigmoid_layers[i].b.get_value()
        
        print >> f, '<biasedlinearity>', W.shape[1], W.shape[0]
        print >> f, '[ ',
        for i in xrange(W.shape[1]):
            for j in xrange(W.shape[0]):
                f.write('%f '%W[j,i])
            if i!=(W.shape[1]-1): print >> f, ' '
        print >> f, ']'
        #bias vector
        print >> f, '[ ',
        for i in xrange(b.shape[0]):
            f.write('%f '%b[i])
        print >> f, ']'
        print >> f, '<sigmoid>', W.shape[1],  W.shape[1]
    
    #logistic regression + softmax
    W = dnn.logLayer.W.get_value()
    b = dnn.logLayer.b.get_value()
    
    print >> f, '<biasedlinearity>', W.shape[1], W.shape[0]
    print >> f, '[ ',
    for i in xrange(W.shape[1]):
        for j in xrange(W.shape[0]):
            f.write('%f '%W[j,i])
        if i!=(W.shape[1]-1): print >> f, ' '
    print >> f, ']'
    #bias vector
    print >> f, '[ ',
    for i in xrange(b.shape[0]):
        f.write('%f '%b[i])
    print >> f, ']'
    print >> f, '<softmax>', W.shape[1], W.shape[1]
    
    f.close()
    
    return None
    