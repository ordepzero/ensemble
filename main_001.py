# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 07:30:16 2018

@author: PeDeNRiQue
"""

import svm as svm
import scipy.io as sio 
import numpy as np

import configuration
import config_enviroment as config_e
import preprocessing as prep
import read_file as rf
import sample
import bases

channels_names = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1",
                  "Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4",
                  "CP6","FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8","F7",
                  "F5","F3","F1","Fz","F2","F4","F6","F8","FT7","FT8",
                  "T7","T8","T9","T10","TP7","TP8","P7","P5","P3","P1",
                  "Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8",
                  "O1","Oz","O2","Iz"]

def load_samples2(conf):
    all_signals = []
    samples = []
    
    file_samples_saved = config_e.create_sample_file(conf)
    #result_file = config_e.create_result_file(conf)
    
    stimulus,codes = prep.get_stimulus_file(conf)

    for i in conf.channels:
        print("Canal: ",i)
        specific_channel = rf.separate_signals([i],conf.size_part,conf.subject,config_e.DIRECTORY_SIGNALS_CHANNELS+"/"," ")
        #print("Filtrando...\t",end="")             
        all_signals.append(prep.filter_decimation(specific_channel,conf))
    
    for i in range(len(stimulus)):
        #for i in range(10):        
            samples_signals = []
            
            for j in range(len(all_signals)):
            #for j in range(10):
                samples_signals.append(all_signals[j][i])
                
            samples.append(sample.Sample(samples_signals,stimulus[i],codes[i]))
            samples_signals = []
            
    return samples

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
    
def load_samples(filename='base.mat'):
    base = sio.loadmat(filename)
    samples = []
    
    for i in range(len(base['signals'])):
        
        concat_signals = base['signals'][i]
        temp_signals = []
        for j in range(0,len(concat_signals),14):
            temp_signals.append(concat_signals[j:j+14])
            
        temp_sample = sample.Sample(temp_signals,base['targets'][0][i],base['codes'][0][i])
    
        samples.append(temp_sample)
    
    return samples

def normalization(signals,mean=[],std=[]):#resulta em um vetor com 896 valores
    
    if(len(std) == 0 and len(mean) == 0 ):
        mean = np.mean(signals, axis=0)
        std = np.std(signals, axis=0)
    
        return (signals - mean)/(std),mean,std
    else:
        return (signals - mean)/(std)


def execute(sel_chan_type,subject,filter_order,highcut): 
    print("Início")
    if(sel_chan_type == 0):
        selected_channels = [1] * 64
    else:
        selected_channels = [0]*64
        
        if(sel_chan_type == 1):
            sct = ['C3', 'CP5', 'CP3', 'F7', 'TP7', 'Pz', 'PO7', 'POz', 'PO8', 'O1', 'Iz']
        else:
            sct = ['Fz','Cz','Pz', 'Oz', 'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8']
        for i in range(len(selected_channels)):
            if(channels_names[i] in sct):
                selected_channels[i] = 1
    
          
    n_bases = 17
    test_base_index = 0
    n_char_base = 5
    n_signals_char = 180
    ciclo_size = 12
    
    config_e.create_directories()    
    
    conf = configuration.Configuration()   
    conf.auto()
    
    conf.subject = subject
    conf.highcut = highcut
    conf.filter_ordem = filter_order
    
    result_file = config_e.create_result_mat_file(conf)
    filename = config_e.create_sample_mat_file(conf)
    
    samples = load_samples(filename)
    
    bases_of_samples = bases.create_sequential_bases(samples,17)
    
    bases_of_samples = bases_of_samples[0:8]
    
    
    train,test = bases.create_train_test_bases(bases_of_samples,selected_channels,test_base_index)
    
    norm_train,mean,std = normalization(train)
    norm_train[:,-1] = train[:,-1]
    
    norm_test = normalization(test,mean,std)
    norm_test[:,-1] = test[:,-1]
    
    
    config = {'subject':conf.subject,
            'lowcut':conf.lowcut,
            'highcut':conf.highcut,
            'filter_order':conf.filter_ordem,
            'selected_channels':selected_channels,
            'test_base_index':test_base_index}
    
    kernel='poly'
    C = 0.5
    coef0 = 1
    gamma = 1
    degree = 2
    params = {'C':C,
              'coef0':coef0,
              'gamma':gamma,
              'degree':degree}
    
    
    all_results = []
    
    if(True):
        c_values = [1e-5,1e-3,1,1e5]
        coef_values = [1e-5,1e-3,1,1e3,1e5]
        gamma_values = [1e-10,1e-3,1,1e3,1e5,1e10]
    else:
        c_values = [1e-10]
        coef_values = [1]
        gamma_values = [1e6]
    
    for C in c_values:
        for coef0 in coef_values:
            for gamma in gamma_values:
            
                params = {'C':C,
                  'coef0':coef0,
                  'gamma':gamma,
                  'degree':degree}
                
                classifier = svm.SVM(kernel,params)
                classifier.train(norm_train)
                train_res,train_prob,train_acc,train_perf = classifier.execute_test(norm_train)
                train_sen = classifier.get_infos()
                
                
                #TESTAR
                
                
                test_res,test_prob,test_acc,test_perf = classifier.execute_test(norm_test)
                test_sen = classifier.get_infos()
                
                results = {'train_res': train_res, 
                               'train_prob': train_prob, 
                               'train_acc': train_acc, 
                               'train_perf': train_perf,
                               'train_sen':train_sen,
                               'test_res':test_res,
                               'test_prob':test_prob,
                               'test_acc':test_acc,
                               'test_perf':test_perf,
                               'test_sen':test_sen,
                               'svm': {'kernel':kernel,'C': C, 'coef0': coef0, 'degree': degree, 'gamma': gamma},
                               'config':config}
                all_results.append(results)
                
                sio.savemat(result_file, {'results':all_results})
                
    for r in all_results:
        print(r['test_perf'])
        
        
if __name__ == "__main__": 
    print("INICIO")
    
    l_sel_chan_type = [0,1,2]
    l_subject = ["A","B"]
    l_filter_order = [5]
    l_highcut = [10,20]
    
    for subject in l_subject:
        for filter_order in l_filter_order:
            for highcut in l_highcut:
                for sel_chan_type in l_sel_chan_type:
    
                    execute(sel_chan_type,subject,filter_order,highcut)