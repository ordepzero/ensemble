# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:14:50 2018

@author: PeDeNRiQue
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:54:01 2018

@author: PeDeNRiQue
"""

import numpy as np
import matlab.engine
import scipy.io as sio
from scipy import signal

import sample
import configuration
import config_enviroment as config_e
import preprocessing as prep
import read_file as rf


def to_filter(signals,lowcut, highcut, fs, order,eng,factor=12.0):    

    aten = 0.5
    
    low = 2 * lowcut / fs
    high = 2 * highcut /fs   
    
    b, a = signal.cheby1(order, aten,[low, high],'bandpass')
        
    filtered_signals = []
    
    for i in range(len(signals)):
            
        temp = signals[i]
           
        filtered_signal = signal.lfilter(b, a, temp)  
        filtered_signal = signal.decimate(filtered_signal,int(factor))
        '''
        double_values = []

        for k in range(len(filtered_signal)):
            double_values.append(float(filtered_signal[k]))            

        filtered_signal = np.array(eng.decimate(matlab.double(double_values),factor))[0]                
        '''   
        filtered_signals.append(filtered_signal)
        
    return filtered_signals

def save_base(samples,conf,prefix=""):
    
    filename = config_e.create_sample_mat_file(conf,prefix)  
    filename = filename
    
    all_signals = []
    all_targets = []
    all_codes = []
    for i in samples:
        all_signals.append(i.get_signals([1]*64))
        all_targets.append(i.target)
        all_codes.append(i.code)
    base = {'signals':all_signals,'targets':all_targets,'codes':all_codes}
    sio.savemat(filename, base)
    
if __name__ == '__main__':
    eng = matlab.engine.start_matlab()
    
    nbOfChars = 85
    n_channels = 64
    n_signals_char = 180
    size_part = 160
    train_or_test = "test"
    
    if(train_or_test == "test"):
        nbOfChars = 100
    else:
        nbOfChars = 85
    
    conf = configuration.Configuration()   
    conf.auto()   
    
    
    for subject in ["A","B"]:
        
        conf.subject = subject
        if(train_or_test == "test"):
            stimulus = prep.get_stimulus_testfile(conf)
            codes = [-1]*len(stimulus)
        else:
            stimulus,codes = prep.get_stimulus_file(conf)
            
        for highcut in [10]:
            for filter_order in [5]:
                for n_char in range(nbOfChars):
                    print(subject,highcut,filter_order,n_char)
                    
                    conf.highcut = highcut
                    conf.filter_ordem = filter_order
                    
                    signal_file = ""
                    if(train_or_test == "test"):
                        if(conf.subject == "A"):
                            signal_file = rf.SIGNAL_TESTFILE_A
                        elif(conf.subject == "B"):
                            signal_file = rf.SIGNAL_TESTFILE_B
                        else:
                            print("Confira o arquivo de sinal do indivíduo.")
                    else:
                        if(conf.subject == "A"):
                            signal_file = rf.SIGNAL_FILE_A
                        elif(conf.subject == "B"):
                            signal_file = rf.SIGNAL_FILE_B
                        else:
                            print("Confira o arquivo de sinal do indivíduo.")
                    print("Separando")
                    sps = rf.separate_signals_parts2(signal_file,n_char,n_channels," ",size_part)
                    print("Filtando")
                    filtred = to_filter(sps,conf.lowcut,conf.highcut,conf.freq,conf.filter_ordem,eng)
                    print("Samples")
                    samples = []
                    cont_stimulus = n_char * n_signals_char
                        
                    for j in range(n_signals_char):
                        
                        samples_signals = []
                        
                        for k in range(n_channels):
                            samples_signals.append(filtred[(k*n_signals_char)+j])
                
                        samples.append(sample.Sample(samples_signals,stimulus[cont_stimulus],codes[cont_stimulus]))
                        cont_stimulus = cont_stimulus + 1
                    print("Salvando")
                    if(train_or_test == "test"):
                        save_base(samples,conf,str(n_char)+"_test_dec_py_")
                    else:
                        save_base(samples,conf,str(n_char)+"_train_dec_py_")