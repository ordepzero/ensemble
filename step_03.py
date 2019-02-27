# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:17:40 2018

@author: PeDeNRiQue
"""


import numpy as np
import matlab.engine
import scipy.io as sio
from scipy import signal
from sklearn.svm import SVC
from sklearn.externals import joblib

import sample
import configuration
import config_enviroment as config_e
import preprocessing as prep
import read_file as rf
import main_002 as m2
import bases
import svm

testA='WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU';
testB='MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR';

matrix=['ABCDEF','GHIJKL','MNOPQR','STUVWX','YZ1234','56789_']



def load_ensemble(conf,prefix=""):
    
    ensembles_filename = config_e.create_ensemble_file(conf,prefix)
    return sio.loadmat(ensembles_filename)
    

def load_samples2(base,size_part=14):
    samples = []
    
    for i in range(len(base['signals'])):
        
        concat_signals = base['signals'][i]
        temp_signals = []
        for j in range(0,len(concat_signals),size_part):
            ts = concat_signals[j:j+size_part]
            if(True):
                low = 2 * 0.1 / 240
                high = 2 * 10 /240 
                
                b, a = signal.cheby1(5, 0.1,[low, high],'bandpass')
                
                #filtered_signal = signal.filtfilt(b, a, xn)      
                ts = signal.lfilter(b, a, ts)
                ts = signal.decimate(ts,12)
            temp_signals.append(ts)
            
        temp_sample = sample.Sample(temp_signals,base['targets'][0][i],base['codes'][0][i])
    
        samples.append(temp_sample)
    
    return samples    


def load_samples(base,size_part=14,n_trials=15):
    samples = []
    #print(len(base['signals']))
    total_trials = 12 * n_trials
    if(total_trials > len(base['signals'])):
        total_trials = len(base['signals'])
        
    for i in range(total_trials):
        
        concat_signals = base['signals'][i]
        temp_signals = []
        for j in range(0,len(concat_signals),size_part):
            ts = concat_signals[j:j+size_part]
            
            temp_signals.append(ts)
            
        temp_sample = sample.Sample(temp_signals,base['targets'][0][i],base['codes'][0][i])
    
        samples.append(temp_sample)
    
    return samples

if __name__ == "__main__": 

    nbOfChars = 100
    
    conf = configuration.Configuration()   
    conf.auto() 
    conf.subject="A"
    conf.filter_ordem=5
    conf.highcut=10 
    n_trials_to_test = 15
    list_n_trials_to_test = range(1,16)
    #ensembles = load_ensemble(conf,"07_26_15_05_25_train_dec_py_at01")# 97%
    #ensembles = load_ensemble(conf,"07_26_21_48_01_train_dec_py_at01")# 94%
    
    ensembles = load_ensemble(conf,"xfc_lda_17b_64c_08_28_19_03_04_train_dec_py_at01")
    #[11, 16, 26, 35, 34, 38, 41, 45, 50, 60, 61, 63, 64, 65, 69] Indivíduo A SVM poly at01 channel_selection
    #[24, 28, 34, 40, 41, 46, 57, 64, 70, 65, 68, 72, 71, 72, 75] Indivíduo B SVM poly at01 channel_selection
    
    # FIltro de ordem 8 ondulação máxima de 0,5 64 canais com 15 sequências
    # [6] A
    # [3] B
    perfs = []
    for cont_list_n_trials_to_test in list_n_trials_to_test:
        n_trials_to_test = cont_list_n_trials_to_test
        cont_hits = 0
        for i in range(nbOfChars):
            prefix = str(i)+"_test_dec_py_at01"
            #prefix = str(i)+"_test_dec_py_"
            filename = config_e.create_sample_mat_file(conf,prefix)
            #print(filename)
            
            signal_file_test = sio.loadmat(filename)
        
            #samples = load_samples(signal_file_test,160)# Estava com este parametro antes
            samples = load_samples(signal_file_test,14,n_trials_to_test)
            print(len(samples))
            
            selected_channels = [1]*64
            test_base = bases.create_base(samples,selected_channels)
            
            cont = 0
            final_results = []
            final_erros = []
            c_values = []
            
            for cont_ensemble in ensembles['ensembles'][0]:
                #print(cont,sum(cont_ensemble['selected_channels'][0][0][0]))
                cont = cont+1
                selected_channels = cont_ensemble['selected_channels'][0][0][0]
                mean = cont_ensemble['mean'][0][0][0]
                std = cont_ensemble['std'][0][0][0]
                
                test_base = bases.create_base(samples,selected_channels)
                norm_test = m2.normalization(test_base,mean,std)
                norm_test[:,-1] = test_base[:,-1]
                
                inputs_test = norm_test[:,:-1]    
                targets_test = norm_test[:,-1]
                
                clf = joblib.load(cont_ensemble['clf_filename'][0][0][0])
                result = clf.predict(inputs_test)
            
                final_results.append(result)
            
            c = [0]*12
            matrix_avaliate = [] #AVALIACAO COM 36
            for result in final_results:
                for a,b in zip(targets_test,result):
                    if(b == 1):
                        c[int(a)-1] = c[int(a)-1]+1
                        #print(a,b)
            
            
            column_index = c[0:6].index(max(c[0:6]))
            line_index = c[6:12].index(max(c[6:12]))
            
            for ii in range(6):
                for jj in range(6):
                    matrix_avaliate.append(c[jj]+c[ii+6])
                    #print(c[ii],c[jj+6],tm[ii][jj])
            
            if(conf.subject == "A"):
                testB = testA
            
            if(matrix[line_index][column_index] == testB[i]):
                cont_hits = cont_hits + 1
                
            ma_ind = matrix_avaliate.index(max(matrix_avaliate))    
            matrix_avaliate_result = matrix[int(ma_ind/6)][ma_ind%6]
            print("Precited word:",matrix[line_index][column_index],testB[i],matrix_avaliate_result,i+1,cont_hits)
            #print(matrix_avaliate.index(max(matrix_avaliate)))
            #break
        perfs.append(cont_hits)
        print(conf.subject,perfs)