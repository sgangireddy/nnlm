'''
Created on 3 Nov 2011

@author: Pawel Swietojanski <P.Swietojanski@sms.ed.ac.uk>
'''

import logging

class LearningRate(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
    def get_rate(self):
        pass
    
    def get_next_rate(self, current_error):
        pass


class LearningRateList(LearningRate):
    
    def __init__(self, learning_rates_list):
        self.lr_list = learning_rates_list
        self.epoch = 1
        
    def get_rate(self):
        if self.epoch <= len(self.lr_list):
            return self.lr_list[self.epoch-1]
        return 0.0
    
    def get_next_rate(self, current_error=None):
        self.epoch += 1
        return self.get_rate()

class LearningRateNewBob(LearningRate):
    """
    Exponential learning rate schema
    """
    
    def __init__(self, start_rate, scale_by, max_epochs, \
                 min_derror_ramp_start, min_derror_stop, init_error, epoch_number, \
                 patience = 0, zero_rate = None, ramping = True):
        """
        :type start_rate: float
        :param start_rate: 
        
        :type scale_by: float
        :param scale_by: 
        
        :type max_epochs: int
        :param max_epochs: 
        
        :type min_error_start: float
        :param min_error_start: 
        
        :type min_error_stop: float
        :param min_error_stop: 
        
        :type init_error: float
        :param init_error: 
               
        """
        self.start_rate = start_rate
        self.init_error = init_error
        self.init_patience = patience
        
        self.rate = start_rate
        self.scale_by = scale_by
        self.max_epochs = max_epochs
        self.min_derror_ramp_start = min_derror_ramp_start
        self.min_derror_stop = min_derror_stop
        self.lowest_error = init_error
        
        self.epoch = epoch_number
        self.ramping = ramping
        self.patience = patience
        self.zero_rate = zero_rate
        
    def reset(self):
        self.rate = self.start_rate
        self.lowest_error = self.init_error
        self.epoch = 1
        self.ramping = False
        self.patience = self.init_patience
    
    def get_rate(self):
        if (self.epoch==1 and self.zero_rate!=None):
            return self.zero_rate
        return self.rate  
    
    def get_next_rate(self, current_error):
        """
        :type current_error: float
        :param current_error: percentage error 
        
        """
        
        diff_error = 0.0
        
        if ( (self.max_epochs > 10000) or (self.epoch >= self.max_epochs) ):
            #logging.debug('Setting rate to 0.0. max_epochs or epoch>=max_epochs')
            self.rate = 0.0
        else:
            diff_error = self.lowest_error - current_error
            
            if (current_error < self.lowest_error):
                self.lowest_error = current_error
    
            if (self.ramping):
                #if (diff_error < self.min_derror_stop):
                #    if (self.patience > 0):
                        #logging.debug('Patience decreased to %f' % self.patience)
                #        self.patience -= 1
                #        self.rate *= self.scale_by
                #    else:
                        #logging.debug('diff_error (%f) < min_derror_stop (%f)' % (diff_error, self.min_derror_stop))
                #        self.rate = 0.0
                #else:
                self.rate *= self.scale_by
		if self.epoch == 15:
		   self.rate = 0.0
            else:
                if (diff_error < self.min_derror_ramp_start):
                    #logging.debug('Start ramping.')
                    self.ramping = True
                    self.rate *= self.scale_by
            
            self.epoch += 1
    
        return self.rate

    
class AdaptiveLearningRate(LearningRate):
    def __init__(self):
        pass
    
    def get_rate(self):
        pass
    
    def get_next_rate(self, dbn):
        pass
