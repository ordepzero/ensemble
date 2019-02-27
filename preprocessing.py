# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:26:39 2018

@author: PeDeNRiQue
"""

import numpy as np

import sample
import read_file as rf
import configuration as config
import config_enviroment as config_e
import filters


def attach_attributes_targets(all_signals,stimulus,conf):
    samples = []    
    
    print("Separando os sinais...\t",end="")
    
    samples_signals = []
    step = len(conf.channels)
    index = 0
    for i in range(0,len(all_signals),step): 
        if(i > len(all_signals)):
            break
        for j in range(i,(i+step)):
            samples_signals.append(all_signals[j])
        samples.append(sample.Sample(samples_signals,stimulus[index]))
        index = index + 1
        samples_signals = []
        
    print("[Ok]")
    
    return samples
    

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
        
def get_stimulus_testfile(conf):
    if(conf.subject == "A"):
        sc = rf.process_stimulus_file(rf.STIMULUS_TESTCODE_A)
        return sc
    elif(conf.subject == "B"):
        sc = rf.process_stimulus_file(rf.STIMULUS_TESTCODE_B)
        return sc
    else:
        print("Erro no carregamento do arquivo do estímulos")
        return None 
    
def get_signals_file(conf,directory):
    signals = []
    
    for i in conf.channels:
        specific_channel = rf.separate_signals([i],conf.size_part,conf.subject,directory," ")
            
        signals.append(specific_channel)
    return signals
    
def get_signals_file2(conf,directory):
    print("Carregando os sinais...\t",end="") 
    signals = []
    #print(conf.channels)
    cont = 0
    for i in conf.channels:
        specific_channel = rf.separate_signals([i],conf.size_part,conf.subject,directory," ")
        #print(str(cont))
        #cont += 1
        signals = signals + specific_channel
    print("[Ok]")
    return signals
    
 
    
def read_samples_saved(filename):
    array = []
    with open(filename,"r") as f:
        for line in f:
            data = line.split()
            array.append(list(map(float, data)))
    values = np.array(array)
    cont = 0
    temp_target = 0
    cont_channel = 0
    signals_temp = []
    samples = []
    for i in range(len(values)):            
        if(cont == 0):
            cont = cont + 1
        elif(cont == 1):
            temp_target = values[i][0]
            cont = cont + 1
        elif(cont == 2):
            cont_channel = cont_channel + 1
            signals_temp.append(values[i])
            if(cont_channel == 64):
                cont_channel = 0
                cont = 0
                samples.append(sample.Sample(signals_temp,temp_target,0))
                signals_temp = []
    return samples
    
def save_samples(file_samples_saved,samples):
    samples_file = open(file_samples_saved,"w+") 
    print("Salvando sinais filtrados...\t",end="")
    for i in range(len(samples)):
        samples_file.write(str(i)+"\n")
        samples_file.write(str(samples[i].target)+"\n")
        
        for j in range(len(samples[i].signals)):
            for k in range(len(samples[i].signals[j])):
                samples_file.write(str(samples[i].signals[j][k])+" ")
            samples_file.write("\n")
    samples_file.close() 
    print("[Ok]")
    
def filter_decimation(signals,conf):
    
    filtered_signals = []
    lowcut = conf.lowcut
    highcut = conf.highcut
    fe = conf.freq
    order = conf.filter_ordem
    aten = conf.aten
    to_decimate = conf.to_decimate
    
    for sig in signals:
        filtered_signals.append(filters.chebyshev1_filter(sig,lowcut,highcut,fe,order,aten,to_decimate,downsampling=conf.downsampling))
        #filtered_signals.append(filters.butter_bandpass_filter(sig,lowcut,highcut,fe,order,to_decimate))
    
    return filtered_signals


def to_normalize(base,each_seg=True):
    new_base = []
    
    if(each_seg):
        max_value = np.matrix(base).max()
        min_value = np.matrix(base).min()
        
        for line in base:
            values = []
            for value in line:
                if(True):            
                    r = (value-min_value)/(max_value-min_value)
                elif(False):
                    r = (2 *(value-min_value)/(max_value-min_value)) - 1
                values.append(r)
            new_base.append(values)
    else:
                
        for line in base:
            max_value = np.matrix(line).max()
            min_value = np.matrix(line).min()
            
            values = []
            for value in line:
                if(False):            
                    r = (value-min_value)/(max_value-min_value)
                elif(True):
                    r = (2 *(value-min_value)/(max_value-min_value)) - 1
                values.append(r)
            new_base.append(values)
    return new_base


def normalize_base(base):
    print("Normalizando...\t",end="") 
    
    new_base = to_normalize(base)    
        
    print("[Ok]")
    return new_base
    
def normalize_each_channel(signals,n_channels):
    print("Normalizando...\t",end="") 
    temp_normalizeds = []
    normalizeds = []
        
    for i in range(n_channels):
        temp = []
        
        for j in range(i,len(signals),n_channels):
            temp.append(signals[j])
        temp_normalizeds.append(to_normalize(temp))
    
    print(len(temp_normalizeds),len(temp_normalizeds[0]))
    
    for i in range(len(temp_normalizeds[0])):
        for j in range(len(temp_normalizeds)):
            normalizeds.append(temp_normalizeds[j][i])
            
    print("[Ok]")
    return normalizeds
    
if __name__ == "__main__":

    conf = config.Configuration()   
    conf.auto()
    
    path = config_e.DIRECTORY_SIGNALS_CHANNELS+"/"
    
    specific_channel = get_signals_file2(conf,path)
