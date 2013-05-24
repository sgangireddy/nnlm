'''
Created on 21 Dec 2011

@author: s1136550
'''

import os, re

class GlobalCfg(object):
    def __init__(self):
        pass
    
    @staticmethod
    def get_working_dir():
        return os.environ['RESULTS_DIR']
    
    @staticmethod
    def get_temp_dir():
        return os.environ['TEMP_DIR']


class FileSystemUtils(object):
    '''
    Various fs utilities
    '''

    def __init__(self):
        pass
    
    @staticmethod
    def ensure_dirs_structure(path):
        dir = os.path.dirname(path)
        if(not os.path.isdir(dir)):
            if (not os.path.isabs(dir)): # makedirs function may be confused with '..' in paths
                dir = os.path.abspath(dir)
            try:    
                os.makedirs(dir)
            except OSError: #probably it's because the dir already exists
                pass
        return None


class PathModifier(object):
    def __init__(self, pattern=None, replace_to=None):
        self.pattern=pattern
        self.replace_to=replace_to
    
    def get_path(self, path):
        if(self.pattern!=None and self.replace_to!=None):
            return re.sub(r''+self.pattern+'', self.replace_to, path, count=1)
        return path
    

        
         