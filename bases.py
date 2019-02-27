# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:24:47 2018

@author: PeDeNRiQue
"""

import numpy as np
from random import randint

def signals_base(samples,selected_channels):
    base = []
    
    for i in range(len(samples)):
        base.append(samples[i].get_concat_signals(selected_channels))   
    return np.array(base)

def create_sequential_bases(samples,n_bases=5):
    bases = []  

    step = int(len(samples)/n_bases)
    
    if(step == 0):
        step = 1
    
    for i in range(0,len(samples),step):
        if(len(bases) < n_bases):
            bases.append(samples[i:(i+step)])
        
    return bases
    
def create_seq_balanced_bases(samples,n_bases=5):
    new_samples = []
    
    for i in range(len(samples)):
        if(samples[i].target == 1):
            new_samples.append(samples[i])
            begin = i - 5
            end = i + 5
            
            if(begin < 0):
                begin = 0
            if(end >= len(samples)):
                end = len(samples) - 2
           
            while(True):
                
                cont = randint(begin, end)
                #print(cont)
                if(cont > 0 and cont < len(samples)-1):
                    if(samples[cont].target != 1):
                        new_samples.append(samples[cont])
                        break
    return create_sequential_bases(new_samples,n_bases)

def create_train_test_bases(bases_of_samples,selected_channels,test_base_index):
    
    i = test_base_index
    
    if(test_base_index < 0):
        i = 0
    elif(test_base_index >= len(bases_of_samples)):
        i = len(bases_of_samples) - 1
    test = []
    train = []
    for j in range(len(bases_of_samples)):
        if(j == i):
            for x in range(len(bases_of_samples[j])):
                train.append(bases_of_samples[j][x].get_concat_signals(selected_channels))
        else:
            for x in range(len(bases_of_samples[j])):
                test.append(bases_of_samples[j][x].get_concat_signals(selected_channels))
                    
    return  np.array(train), np.array(test)



def create_base(samples,selected_channels):
    result_base = []
    
    
    for s in samples:
        result_base.append(s.get_concat_signals(selected_channels))
        
    return np.array(result_base)
    
    
def create_code_base(samples):
    result_base = []
    
    
    for s in samples:
        result_base.append(s.code)
        
    return np.array(result_base)
    
    
    
    
    
    
    
    
    
    
    
    