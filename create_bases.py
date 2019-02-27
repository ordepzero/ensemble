# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:49:03 2018

@author: PeDeNRiQue
"""
import matlab.engine
import configuration as config
import config_enviroment as config_e
import numpy as np
import read_file as rf
import scipy
import scipy.io as sio 
import sample

def chebyshev1_filter(xn,lowcut, highcut, fs, order,aten,_decimate,downsampling):
    #nyq = 0.5 * fs
    #low = lowcut / nyq
    #high = highcut / nyq
    
    low = 2 * lowcut / fs
    high = 2 * highcut /fs   
    
    b, a = scipy.signal.cheby1(order, aten,[low, high],'bandpass')
    
    #filtered_signal = signal.filtfilt(b, a, xn)      
    filtered_signal = scipy.signal.lfilter(b, a, xn)  
    
    double_values = []
    
    for k in range(len(filtered_signal)):
        double_values.append(float(filtered_signal[k]))
    
    #print(">>>",downsampling)
    if(True):
        #filtered_signal = decimate(filtered_signal,int(downsampling))
        #filtered_signal = signal.decimate(filtered_signal,int(fs/highcut)) #Fator
        #filtered_signal = signal.resample(filtered_signal,10) #Numero de amostras
        #filtered_signal = signal.decimate(filtered_signal,int(downsampling)) #Fator
        filtered_signal = np.array(eng.decimate(matlab.double(double_values),12.0))[0]
    return filtered_signal

def get_stimulus_file(conf):
    if(conf.subject == "A"):
        sf = rf.process_stimulus_type_file(rf.STIMULUS_FILE_A)
        sc = rf.process_stimulus_file(rf.STIMULUS_CODE_A)
        return sf,sc
    elif(conf.subject == "B"):
        sf = rf.process_stimulus_type_file(rf.STIMULUS_FILE_B)
        sc = rf.process_stimulus_file(rf.STIMULUS_CODE_B)
        return sf,sc
    else:
        print("Erro no carregamento do arquivo do estímulos")
     
        return None

def save_base(samples,conf):
    
    filename = config_e.create_sample_mat_file(conf)  
    
    
    all_signals = []
    all_targets = []
    all_codes = []
    for i in samples:
        all_signals.append(i.get_signals([1]*64))
        all_targets.append(i.target)
        all_codes.append(i.code)
    base = {'signals':all_signals,'targets':all_targets,'codes':all_codes}
    sio.savemat(filename, base)    
    
    

if __name__ == "__main__": 
    
    eng = matlab.engine.start_matlab()
    
    config_e.create_directories()    
    
    conf = config.Configuration()   
    conf.auto()
    
    result_file = config_e.create_result_file(conf)
    
    stimulus,codes = get_stimulus_file(conf)
    for sub in ["A","B"]:
        for ch in [30]:
            for corder in [4,5,8]:
                print(ch,corder)
                conf.highcut = ch
                conf.filter_ordem = corder
                conf.subject = sub
                
                all_signals = []
                samples = []
        
                for i in conf.channels:
                    print("Canal: ",i)
                    specific_channel = rf.separate_signals([i],conf.size_part,conf.subject,config_e.DIRECTORY_SIGNALS_CHANNELS+"/"," ")
                    #print("Filtrando...\t",end="")  
                    filtered_signals = []
                    lowcut = conf.lowcut
                    highcut = conf.highcut
                    fe = conf.freq
                    order = conf.filter_ordem
                    aten = conf.aten
                    to_decimate = conf.to_decimate       
                    downsampling = conf.downsampling
                    
                    temp_i = []
                    for sch in specific_channel:
                        temp_i.append(chebyshev1_filter(sch,lowcut,highcut,fe,order,aten,to_decimate,downsampling))
                    
                    all_signals.append(temp_i)#64x15300
                    
                for st in range(len(stimulus)):#15300
                    #for i in range(10):        
                    samples_signals = []
                    
                    for j in range(len(all_signals)):#64
                    #for j in range(10):
                        samples_signals.append(all_signals[j][st])
                        
                    samples.append(sample.Sample(samples_signals,stimulus[st],codes[st]))
                    samples_signals = []
                            
                save_base(samples,conf)
        
    
    
    
    