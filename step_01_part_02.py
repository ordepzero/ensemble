# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 09:38:01 2018

@author: PeDeNRiQue
"""

import copy
import pickle
from sklearn.svm import SVC
import scipy.io as sio 
import numpy as np
from sklearn.externals import joblib
import os
import datetime

import configuration
import config_enviroment as config_e
import preprocessing as prep
import read_file as rf
import sample
import bases
import svm

channels_names = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1",
                  "Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4",
                  "CP6","FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8","F7",
                  "F5","F3","F1","Fz","F2","F4","F6","F8","FT7","FT8",
                  "T7","T8","T9","T10","TP7","TP8","P7","P5","P3","P1",
                  "Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8",
                  "O1","Oz","O2","Iz"]

def create_result_dic():
    results = {'train_res': [], 
               'train_prob': [], 
               'train_acc': [], 
               'train_perf': [],
               'train_sen':[],
               'test_res':[],
               'test_prob':[],
               'test_acc':[],
               'test_perf':[],
               'test_sen':[],
               'algoritmo':[],
               'kernel':[],
               'C': [], 
               'coef0': [], 
               'degree': [], 
               'gamma': [],
               'subject':[],
               'lowcut':[],
               'highcut':[],
               'filter_order':[],
               'selected_channels':[],
               'test_base_index':[],
               'mean':[],
               'std':[],
               #'support_':[],
               #'support_vectors_':[],
               #'n_support_':[],
               #'dual_coef_':[],
               #'coef_':[],
               #'intercept_':[],
               'clf_filename':[],
               'ch_perfs':[],# performances indiduais dos canais 
               'all_perfs':[],
               'all_keep_chan':[]}

    return results


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


def load_all_samples(config_e,conf,n_all_char,prefix=""):
    samples = []
    for i in range(n_all_char):
        filename = config_e.create_sample_mat_file(conf,str(i)+prefix)
        #print(filename)
        if(len(samples) == 0):
            samples = load_samples(filename)
        else:
            samples = samples+load_samples(filename)
    return samples

def create_svm(kernel,params,norm_train):
    c  = params['C']
    co = params['coef0']
    de = params['degree']
    gm = params['gamma']
    
    clf = SVC(kernel='linear',probability=True,C=c,coef0=co,degree=de,gamma=gm)
    inputs_train = norm_train[:,:-1]
    targets_train = norm_train[:,-1]
    clf.fit(inputs_train,targets_train)    
    return clf

def calculate_performance(results,targets):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(results)):
        if(results[i] == targets[i]):
            
            if(results[i] != 1):
                tn = tn + 1
            elif(results[i] == 1):
                tp = tp + 1
        elif(results[i] != 1):
            fn = fn + 1
        elif(results[i] == 1):
            fp = fp + 1
    perf = tp / (tp+fp+fn)
    acu = tp / (tp+tn+fp+fn)
    return perf,acu,tp,tn,fp,fn

def test_classfier(clf,base):
    inputs = base[:,:-1]
    targets = base[:,-1]
    
    results = clf.predict(inputs)
    results_prob = clf.predict_proba(inputs)
    return calculate_performance(results,targets),results,results_prob
    
def execute_svm_train(bases_of_samples,selected_channels,degree,conf,first_index,subdic_name):

    n_total_channels = 64
    results = create_result_dic()
    
    for test_base_index in range(len(bases_of_samples)):
        
        print("Indice da base: "+str(first_index+test_base_index))
        kernel = 'linear'  
    
        
        if(False):
            c_values = [0.01,0.1,0.5,1 ]
            coef_values = [1e-5,1e-3,1]
            gamma_values = ["auto",1e-3,1]
        else:
            c_values = [1]
            coef_values = [0]
            gamma_values = ['auto']
        
        result_file = config_e.create_result_mat_file_subdic(conf,subdic_name,str(test_base_index+first_index)+"_")
        pkl_filename = config_e.create_result_pkl_file_subdic(conf,test_base_index+first_index,subdic_name)

        current_test_perf = 0
        current_classifier = None
        for C in c_values:
            for coef0 in coef_values:
                for gamma in gamma_values:
                    
                    params = {'C':C,
                      'coef0':coef0,
                      'gamma':gamma,
                      'degree':degree}
                    
                    
                    max_channels_selected = 50
                    n_channels_to_add = 6
                    keep_selected_channels = [0]*n_total_channels
                    
                    sct = ['PO7','PO8']
                    control_sct = False

                    for cont_ksc in range(len(keep_selected_channels)):
                        if(channels_names[cont_ksc] in sct):
                            keep_selected_channels[cont_ksc] = 1
                    
                    individual_perfs = [-1]*64
                    all_perfs_after_selec_n_char = [] # Após a seleção de N canais é realizado a avaliação e é salva aqui
                    all_keep_chan = []
                    while(sum(keep_selected_channels) < max_channels_selected):
                        perfs_each_channel = [0]*n_total_channels
                        
                        print("N Channels selecteds:"+str(sum(keep_selected_channels)))
                        if(control_sct):
                            for cont_channel in range(n_total_channels):
                                selected_channels = keep_selected_channels
                                if(keep_selected_channels[cont_channel] == 0):
                                    print(cont_channel,end=" ")
                                    selected_channels[cont_channel] = 1
                                
                                    train,test = bases.create_train_test_bases(bases_of_samples,selected_channels,test_base_index)
                        
                                    norm_train,mean,std = normalization(train)
                                    norm_train[:,-1] = train[:,-1]
                                    
                                    norm_test = normalization(test,mean,std)
                                    norm_test[:,-1] = test[:,-1]
                                    
                       
                                    clf = create_svm(kernel,params,norm_train)
                                    perfs = test_classfier(clf,norm_train)
                                    train_perf = perfs[0][0]
                                    train_acc = perfs[0][1]
                                    train_sen = perfs[0][2:]
                                    train_prob = perfs[2]
                                    train_res = perfs[1]
                                    #print(clf)
                                    #break
                                    #TESTAR
                                    
                                    perfs = test_classfier(clf,norm_test)
                                    test_perf = perfs[0][0]
                                    test_acc = perfs[0][1]
                                    test_sen = perfs[0][2:]
                                    test_prob = perfs[2]
                                    test_res = perfs[1]
                                    
                                    selected_channels[cont_channel] = 0
                                    perfs_each_channel[cont_channel] = test_perf # Guarda a performance após adição do canal
                                    
                                if(individual_perfs[cont_channel] == -1):
                                    individual_perfs[cont_channel] = test_perf
                            # FIM for cont_channel in range(n_total_channels):
                            for cont_channels_to_add in range(n_channels_to_add):
                                while(True):
                                    temp_index_perfs = perfs_each_channel.index(max(perfs_each_channel))
                                    perfs_each_channel[temp_index_perfs] = -1
                                    if(keep_selected_channels[temp_index_perfs] == 0):                                    
                                        keep_selected_channels[temp_index_perfs] = 1
                                        break
                        else:                            
                            control_sct = True
                            
                        # TESTAR A NOVA COMBINACAO DE CANAIS 
                        # print("AVALIAR NOVA COMBINACAO")
                        selected_channels = keep_selected_channels  
                        train,test = bases.create_train_test_bases(bases_of_samples,selected_channels,test_base_index)
                    
                        norm_train,mean,std = normalization(train)
                        norm_train[:,-1] = train[:,-1]
                        
                        norm_test = normalization(test,mean,std)
                        norm_test[:,-1] = test[:,-1]
                        
           
                        clf = create_svm(kernel,params,norm_train)
                        perfs = test_classfier(clf,norm_train)
                        train_perf = perfs[0][0]
                        train_acc = perfs[0][1]
                        train_sen = perfs[0][2:]
                        train_prob = perfs[2]
                        train_res = perfs[1]
                        #print(clf)
                        #break
                        #TESTAR
                        
                        perfs = test_classfier(clf,norm_test)
                        test_perf = perfs[0][0]
                        test_acc = perfs[0][1]
                        test_sen = perfs[0][2:]
                        test_prob = perfs[2]
                        test_res = perfs[1]
                        
                        all_perfs_after_selec_n_char.append(test_perf)
                        all_keep_chan.append(copy.copy(selected_channels))
                        print("<",sum(selected_channels),test_perf,all_perfs_after_selec_n_char,">")
                        if(test_perf >= current_test_perf):
                            current_test_perf = test_perf
                            current_classifier = clf
                            
                            results['train_res'] = train_res
                            results['train_prob'] = train_prob
                            results['train_acc'] = train_acc
                            results['train_perf'] = train_perf
                            results['train_sen'] = train_sen
                            results['test_res'] = test_res
                            results['test_prob'] = test_prob
                            results['test_acc'] = test_acc
                            results['test_perf'] = test_perf
                            results['test_sen'] = test_sen
                            results['algoritmo'] = "svm"
                            results['kernel'] = kernel
                            results['C'] = C
                            results['coef0'] = coef0
                            results['degree'] = degree
                            results['gamma'] = gamma
                            results['subject'] = conf.subject
                            results['lowcut'] = conf.lowcut
                            results['highcut'] = conf.highcut
                            results['filter_order'] = conf.filter_ordem
                            results['selected_channels'] = selected_channels
                            results['test_base_index'] = test_base_index+first_index
                            results['mean'] = mean
                            results['std'] = std
                            results['clf_filename'] = pkl_filename
                            results['ch_perfs'] = individual_perfs# performances indiduais dos canais 
                            results['all_perfs'] = all_perfs_after_selec_n_char
                            results['all_keep_chan'] = all_keep_chan
                        
        joblib.dump(current_classifier, pkl_filename) 
        sio.savemat(result_file, results)
    
def execute(sel_chan_type,subject,filter_order,highcut,degree=2,subdic_name="",prefix=""): 
    #print("Início")
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
    n_all_char = 85
    
    config_e.create_directories()    
    
    conf = configuration.Configuration()   
    conf.auto()
    
    conf.subject = subject
    conf.highcut = highcut
    conf.filter_ordem = filter_order
    
    result_file = config_e.create_result_mat_file(conf)
    
    samples = load_all_samples(config_e,conf,n_all_char,prefix)
    
    
    bases_of_samples = bases.create_sequential_bases(samples,n_bases)
    
    first_index = 8
    execute_svm_train(bases_of_samples[0:first_index],selected_channels,degree,conf,0,subdic_name)
    execute_svm_train(bases_of_samples[first_index:],selected_channels,degree,conf,first_index,subdic_name)
    
                    
    
#cs4f indica a seleção de canais com 4 fixados
#def temp_main():       
if __name__ == "__main__": 
    print("INICIO")
    
    l_sel_chan_type = [0]
    l_subject = ["A"]
    l_filter_order = [5]
    l_highcut = [10]
    l_degree = [3]
    
    t = str(datetime.datetime.now()) 
    subdic_name = t.replace("-","_").replace(":","_").replace(" ","_")[:19]
    prefix = "_train_dec_py_at01"
    
    subdic_name = "cs4f_5b"+subdic_name[5:]+prefix
    for subject in l_subject:
        for filter_order in l_filter_order:
            for highcut in l_highcut:
                for sel_chan_type in l_sel_chan_type:
                    for degree in l_degree:
                        print(subject,filter_order,highcut,sel_chan_type,degree)
                        execute(sel_chan_type,subject,filter_order,highcut,degree,subdic_name,prefix) # O 0 indica da sub base 0 a 8
                        
#if __name__ == "__main__":                    
def test_load_samples():
    
    
    config_e.create_directories()    
    
    conf = configuration.Configuration()   
    conf.auto()
    
    conf.subject = "B"
    conf.highcut = "10"
    conf.filter_ordem = 5
    
    result_file = config_e.create_result_mat_file(conf)
    filename = config_e.create_sample_mat_file(conf,"0_train_")
    #print(filename)
    samples = load_samples(filename)        
                    
                    
                    