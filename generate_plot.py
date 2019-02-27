# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:20:50 2018

@author: PeDeNRiQue
"""
import numpy as np
import matplotlib.pyplot as plt
import configuration
import config_enviroment as config_e
import step_01

prefix = "_train_dec_py_at01"
#prefix = "_train_dec_py_"
n_all_char = 85
config_e.create_directories()    
    
conf = configuration.Configuration()   
conf.auto()

selected_channels = [0]*64
selected_channels[57]=1

samples = step_01.load_all_samples(config_e,conf,n_all_char,prefix)
#samples = samples[:180]
target_N1 = [0]*14
target_P1 = [0]*14

cont_P1 = 0
cont_N1 = 0
for cont_sample in samples:
    if(cont_sample.target == 1):
        cont_P1 = cont_P1 + 1
        target_P1 = np.add(target_P1,cont_sample.get_concat_signals(selected_channels)[0:14])
    else:
        cont_N1 = cont_N1 + 1
        target_N1 = np.add(target_N1,cont_sample.get_concat_signals(selected_channels)[0:14])
        


target_P1 = target_P1/cont_P1
target_N1 = target_N1/cont_N1

plt.plot(target_N1)
plt.plot(target_P1)
plt.legend(['Sem P300','Com P300'])
plt.axis([1, 13, -1, 3])
plt.xlabel('Tempo (amostras)')
#plt.ylabel('Amplitude '+'(\u03bc'+'V)')
plt.ylabel('Amplitude (mV)')
plt.show()