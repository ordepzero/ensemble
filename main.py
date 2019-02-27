# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:21:25 2018

@author: PeDeNRiQue
"""

import os
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from random import randint

import datetime
from scipy import signal
import matplotlib.pyplot as plt
import csv
import operator

import read_file as rf
import neural_network as nn
import svm as svm
import linear_discriminant_analysis as lda
import filters
import configuration as config
import config_enviroment as config_e
import sample
import preprocessing as prep
import bases
import mmspg_set as mmspg


'''
{'C': 0.1, 'coef0': 1, 'degree': 3, 'gamma': 0.5}
{'C': 1.0, 'coef0': 1, 'degree': 2, 'gamma': 1}
{'C': 0.01, 'coef0': 0, 'degree': 3, 'gamma': 2}
{'C': 0.5, 'coef0': 5, 'degree': 2, 'gamma': 2}
{'C': 0.1, 'coef0': 5, 'degree': 2, 'gamma': 2}
{'C': 1.0, 'coef0': 0, 'degree': 2, 'gamma': 1}
{'C': 1.0, 'coef0': 0, 'degree': 2, 'gamma': 1} 
{'C': 1.0, 'coef0': 0, 'degree': 2, 'gamma': 1}
{'C': 5.0, 'coef0': 1, 'degree': 2, 'gamma': 0.5}
{'C': 0.01, 'coef0': 1, 'degree': 3, 'gamma': 1}
{'C': 0.5, 'coef0': 1, 'degree': 2, 'gamma': 1}
'''


channels_names = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1",
                  "Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4",
                  "CP6","FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8","F7",
                  "F5","F3","F1","Fz","F2","F4","F6","F8","FT7","FT8",
                  "T7","T8","T9","T10","TP7","TP8","P7","P5","P3","P1",
                  "Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8",
                  "O1","Oz","O2","Iz"]
 


    
def load_samples(conf):   
    
    all_signals = []
    samples = []
    
    file_samples_saved = config_e.create_sample_file(conf)
    #result_file = config_e.create_result_file(conf)
    
    
    if not os.path.exists(file_samples_saved): 
        
        all_signals = prep.get_signals_file(conf,config_e.DIRECTORY_SIGNALS_CHANNELS+"/")
        
        
        if(conf.to_filter):                   
            all_signals = prep.filter_decimation(all_signals,conf)                          
        if(True):
            #all_signals = prep.normalize_base(all_signals)        
            all_signals = prep.normalize_each_channel(all_signals,len(conf.channels))
        stimulus = prep.get_stimulus_file(conf)
        
        samples = prep.attach_attributes_targets(all_signals,stimulus,conf)       
        
        
       #os.makedirs(DIRECTORY_SIGNALS_CHANNELS+"/samples_saved.txt")
        if(True):        
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
            
    else:
        samples = prep.read_samples_saved(file_samples_saved)
        
    return samples
    

def divide_base_and_execute(bases_of_samples,selected_channels,fo,alg,params):
    test = []
    train = []
    med_train_error = 0
    med_test_error = 0
    performances = []
    
    fo.write("Folds: "+str(len(bases_of_samples))+"\n")   
    fo.write("Canais: ")
    channels_s = ""
    for sc in range(len(selected_channels)):
        if(selected_channels[sc] == 1):
            channels_s = channels_s+ " " +channels_names[sc]
            fo.write(channels_names[sc]+" ")
            #fo.write(str(selected_channels[sc])+" ")
    fo.write("\n")    
    print(channels_s)
    for i in range(len(bases_of_samples)):
        test = []
        train = []
        for j in range(len(bases_of_samples)):
            if(j != i):
                for x in range(len(bases_of_samples[j])):
                    train.append(bases_of_samples[j][x].get_concat_signals(selected_channels))
            else:
                for x in range(len(bases_of_samples[j])):
                    test.append(bases_of_samples[j][x].get_concat_signals(selected_channels))
        train_error = 0
        test_error = 0
        
        train = np.array(train)
        test  = np.array(test)
        
        
        if(len(train) == 0 or len(test) == 0):
            break
        if(True):
            if(alg=="rna"):
                if(True):    
                    print("Executando RNA, Opcao 1")
                    train_error,test_error = nn.execute_rna(train,test,fo)
                elif(False):
                    print("Executando RNA, Opcao 2")
                    train_error,test_error = nn.execute_rna(train,test,fo,n_neuron=160,epoch=400)
            elif(alg=="lda"):
                #fo.write("Algoritmo: LDA\n")
                alg_lda = lda.LDA()
                train_error,test_error = alg_lda.execute_lda(train,test)
                mc = alg_lda.get_infos()
                print(mc)
            elif(alg=="svm"):
                #print(str(params))
                #if(params != None):
                #     fo.write(str(params)+"\n")
                alg_svm = svm.SVM()
                train_error,test_error = alg_svm.execute_svm(train,test,'poly',params)
                mc = alg_svm.get_infos()
                print(mc)
                performances.append(alg_svm.calculate_performance())
        
        fo.write("AcTreino:"+str(train_error)+"\tAcTeste:"+str(test_error)+"\n")
        for m in range(len(mc)):
            if(m < len(mc)-1):
                fo.write(str(mc[m])+" ")  
            else:
                fo.write(str(mc[m])+"\n")
        med_train_error += train_error
        med_test_error += test_error
        
        print(str(i)+" "+str(train_error)+" "+str(test_error))
        
    med_train_error = med_train_error/len(bases_of_samples)
    med_test_error = med_test_error/len(bases_of_samples)
    fo.write("AcMedioTreino: "+str(med_train_error)+"\tAcMedioTeste:"+str(med_test_error)+"\n")

    if(len(performances) > 0):
        return np.mean(performances)
    return med_test_error
    
    
def channel_selection(bases_of_samples,selected_channels,fo,alg,params,result1=None):
    begin = 0
    channels1 = copy.copy(selected_channels)
    if(result1 == None):
        if(sum(selected_channels) == 0):
            for i in range(len(selected_channels)):
                if(selected_channels[i] == 0):
                    selected_channels[i] = 1
                    #result1 = to_evaluate(bases_of_samples,selected_channels)
                    result1 = divide_base_and_execute(bases_of_samples,selected_channels,fo,alg,params)

                    selected_channels[i] = 0
                    begin = i
                    break
        else:
            #result1 = to_evaluate(bases_of_samples,selected_channels)
            result1 = divide_base_and_execute(bases_of_samples,selected_channels,fo,alg,params)


    for i in range(begin,len(selected_channels)):
        if(selected_channels[i] == 0):
            selected_channels[i] = 1
            #result = to_evaluate(bases_of_samples,selected_channels)
            result = divide_base_and_execute(bases_of_samples,selected_channels,fo,alg,params)
            if(result > result1):
                result1 = result
                channels1 = copy.copy(selected_channels)
            selected_channels[i] = 0

    if(sum(channels1) >= 32 or sum(channels1) == sum(selected_channels)):
        return channels1,result1
    else:
        return channel_selection(bases_of_samples,channels1,fo,alg,params,result1)


#if __name__ == "__main__":     
def load_mmspg_set(conf):
     
    
    all_signals = []
    samples = []
    
    file_samples_saved = config_e.create_sample_file(conf)
    
    mmspg_set = mmspg.load_set(conf)
    
    #34 138 2 1366/1
    if not os.path.exists(file_samples_saved): 
        print("Preprocessamento...")  
        for i in range(len(mmspg_set)):
            temp_singals = []
            for j in range(len(mmspg_set[i])):
                temp_singals.append(mmspg_set[i][j][0])
            
            filtered = prep.filter_decimation(temp_singals,conf)
            
            normalized = prep.to_normalize(filtered)    
            #print("[Ok]")            
            all_signals.append(normalized)
               
        for i in range(len(all_signals[0])):
            #for i in range(10):        
            samples_signals = []
            
            for j in range(len(all_signals)):
            #for j in range(10):
                samples_signals.append(all_signals[j][i])
                
            samples.append(sample.Sample(samples_signals,mmspg_set[0][i][1]))
            samples_signals = []
        if(False):
            prep.save_samples(file_samples_saved,samples)
        
    else:
        samples = prep.read_samples_saved(file_samples_saved)
        
    return samples


    
def load_samples2(conf):
                
    all_signals = []
    samples = []
    
    file_samples_saved = config_e.create_sample_file(conf)
    #result_file = config_e.create_result_file(conf)
    
    stimulus,codes = prep.get_stimulus_file(conf)
    
    if not os.path.exists(file_samples_saved): 
        print("Preprocessamento...")  
        for i in conf.channels:
            print("Canal: ",i)
            specific_channel = rf.separate_signals([i],conf.size_part,conf.subject,config_e.DIRECTORY_SIGNALS_CHANNELS+"/"," ")
            #print("Filtrando...\t",end="")             
            filtered = prep.filter_decimation(specific_channel,conf) 
            #print("[Ok]")               
            #print("Normalizando...\t",end="") 
            normalized = prep.to_normalize(filtered,False)    
            #print("[Ok]")            
            all_signals.append(normalized)
        print("[Ok]")
            
        #stimulus,codes = prep.get_stimulus_file(conf)
        
        #samples = prep.attach_attributes_targets(all_signals,stimulus,conf)
        for i in range(len(stimulus)):
        #for i in range(10):        
            samples_signals = []
            
            for j in range(len(all_signals)):
            #for j in range(10):
                samples_signals.append(all_signals[j][i])
                
            samples.append(sample.Sample(samples_signals,stimulus[i],codes[i]))
            samples_signals = []
            
        if(True):
            prep.save_samples(file_samples_saved,samples)
        
    else:
        samples = prep.read_samples_saved(file_samples_saved)
        for s,c in zip(samples,codes):
            s.code = c
    return samples

if __name__ == "__main__":
    
    
    n_bases = 5
    test_base_index = 0
    n_char_base = 17
    n_signals_char = 180
    ciclo_size = 12
    
    config_e.create_directories()    
    
    conf = config.Configuration()   
    conf.auto()
    
    result_file = config_e.create_result_file(conf)
                
    samples = load_samples2(conf)
    
    selected_channels = [0]*64
    #selected_channels[60] = 1
    sct = ['C3', 'CP5', 'CP3', 'F7', 'TP7', 'Pz', 'PO7', 'POz', 'PO8', 'O1', 'Iz']
    
    for i in range(len(selected_channels)):
        if(channels_names[i] in sct):
            selected_channels[i] = 1
    
    #base_of_samples = bases.create_sequential_bases(samples)
    
    base_of_samples = bases.create_seq_balanced_bases(samples)
    
    train,test = bases.create_train_test_bases(base_of_samples,selected_channels,test_base_index)
            
    alg_lda = lda.LDA()
    lda = alg_lda.train(train)
    
    count_probas = [0] * 12
    #for j in range(1):
    for j in range(n_char_base):
        char_index = (test_base_index * n_signals_char *  n_char_base) + (j * n_signals_char) 
        test_base = bases.create_base(samples[char_index:char_index+n_signals_char],selected_channels)
        result_char_proba = alg_lda.test_proba(test_base)
        
        for i in range(0,len(result_char_proba)):
            #print(result_char_proba[i],int(samples[char_index+i].code),samples[char_index+i].target)
            #if(result_char_proba[i][1] > 0.8):
            temp_index = int(samples[char_index+i].code) - 1
            count_probas[temp_index] = count_probas[temp_index] + result_char_proba[i][1]
           
        temp_samples = samples[char_index:char_index+ciclo_size]
        line_colunm_foced = []
        for ts in temp_samples:
            if(ts.target == 1):
                line_colunm_foced.append(int(ts.code))
        print(line_colunm_foced)                                                    
        #print(count_probas)
        max_index1, max_value1 = max(enumerate(count_probas[0:6]), key=operator.itemgetter(1))
        max_index2, max_value2 = max(enumerate(count_probas[6:12]), key=operator.itemgetter(1))
        print("[",max_index1+1,max_index2+7,"]")                                                                                
        #print(count_probas[0:6],count_probas[6:12])
    '''
    count_hits = 0
    for j in range(1,17):# 
        char_index = 180 * j
        test_base = bases.create_base(samples[char_index:char_index+180],selected_channels)
        r = alg_lda.test_proba(test_base)
        
        
        correct_codes = []
        for i in range(0,len(r),12):
            temp_r = ""
            if(sum(r[i:i+12]) < 5):
                for s in samples[i+char_index:i+12+char_index]:
                    if(s.target == 1):
                        temp_r = temp_r+str(int(s.code))+" "
                        correct_codes.append(int(s.code))
            #print(r[i:i+12])
        #print(temp_r)
            
        
        codes_count = [0] * 12
        for i in range(len(r)):
            if(r[i] == 1):
                c_code = int(samples[i+char_index].code)-1
                codes_count[c_code] = codes_count[c_code] + 1
        #print(codes_count)
        
        max_index1, max_value1 = max(enumerate(codes_count[0:6]), key=operator.itemgetter(1))
        max_index2, max_value2 = max(enumerate(codes_count[6:12]), key=operator.itemgetter(1))
            
        if((max_index1+1) in correct_codes and (max_index2+1) in correct_codes):
            count_hits = count_hits + 1
    print(count_hits)
    '''
    
    
    
#Trabalhando com a base MMSPG
def test_mmspg():
#if __name__ == "__main__":   
    config_e.create_directories()    
    
    conf = config.Configuration()   
    conf.auto()
    
    samples = load_mmspg_set(conf)
    
    base_of_samples = bases.create_seq_balanced_bases(samples)
    
    selected_channels = [0]*34
    
    result_file = config_e.create_result_file(conf)
    fo = open(result_file,"w+") 
    fo.write(conf.get_info())
                
    channels1,result1 = channel_selection(base_of_samples,selected_channels,fo,conf.alg,conf.params)

            
#if __name__ == "__main__":     
def teste():  
    
    
    config_e.create_directories()    
    
    conf = config.Configuration()   
    conf.auto()
    
    
    for s in ["A"]:
        for i in [10.0,20.0]:
            for j in [5,8]:
        
                conf.filter_ordem = j
                conf.highcut = i
                conf.subject = s
                 
                result_file = config_e.create_result_file(conf)
                
                samples = load_samples2(conf)
            
                
                base_of_samples = bases.create_seq_balanced_bases(samples)
            
                
                selected_channels = [0]*64
            
                fo = open(result_file,"w+") 
                fo.write(conf.get_info())
                
                
                channels1,result1 = channel_selection(base_of_samples,selected_channels,fo,conf.alg,conf.params)
    
                print(channels1,result1)
                fo.close()
            
    '''        
    classifiers = []
    basess = []
    for b in range(len(base)):
        temp_b = bases.signals_base(base[b],selected_channels)        
        basess.append(temp_b)
        
        temp_svm = svm.SVM()
        temp_svm.train(temp_b)
        classifiers.append(temp_svm)
        
        
    for c in range(len(classifiers)):        
        for b in range(len(base)):
            
            if(c != b):
                print("Base:",str(b),"  Classificador:",str(c),end=" ")
                error = classifiers[c].test(basess[b])
                print(str(error))
    '''

    