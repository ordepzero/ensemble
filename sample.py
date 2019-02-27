# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 01:19:59 2018

@author: PeDeNRiQue
"""

import numpy as np

#Cada Sample possui 64 canais
class Sample:
    
    signals = None
    target  = None
    code    = None
    
    def __init__(self,signals,target,code) :
        
        if(target == 0):
            target = -1
        self.target = int(target)
        self.signals = np.array(signals)
        self.code   = int(code)
        
    def get_concat_signals(self,selected_channels):
        concat_signals = []        
        
        for i in range(len(self.signals)):
            if(selected_channels[i] == 1):
                for j in range(len(self.signals[i])):
                    concat_signals.append(self.signals[i][j])
            
        concat_signals.append(self.target)
        return concat_signals
    def get_signals(self,selected_channels):
        concat_signals = []        
        
        for i in range(len(self.signals)):
            if(selected_channels[i] == 1):
                for j in range(len(self.signals[i])):
                    concat_signals.append(self.signals[i][j])
        return concat_signals
    
    