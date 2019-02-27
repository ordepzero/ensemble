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
    
def load_ensemble2(conf,result_filename,minimun_avaliation):
    
    minimun_avaliation = str(0.21).replace(".","")

    #result_filename = "cs4f_08_10_00_33_06_train_dec_py_at01"
    result_directory = config_e.DIRECTORY_BASE+"/results_"+conf.subject+"/"+result_filename
    ensemble_sett = sio.loadmat(result_directory+"/"+minimun_avaliation+".mat")
    
    ensembles2 = []
    for cont_ensemble in range(17):
        temp_ensemble = joblib.load(result_directory+"/"+str(cont_ensemble)+"_"+minimun_avaliation+".pkl")
        ensembles2.append(temp_ensemble)
    

    return ensembles2,ensemble_sett
        
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
    
    result_filename = "cs4f_1b_svm_09_07_09_37_35_train_dec_py_at01"
    TYPE = False
    if(TYPE):
        minimun_avaliation = 0.25
        ensembles,ensemble_sett = load_ensemble2(conf,result_filename,minimun_avaliation)        
        all_ensembles = ensembles
    else:
        ensembles = load_ensemble(conf,result_filename)
        all_ensembles = ensembles['ensembles'][0]
        
    perfs = []
    for cont_list_n_trials_to_test in list_n_trials_to_test:
        n_trials_to_test = cont_list_n_trials_to_test
        cont_hits = 0
        for i in range(nbOfChars):
            prefix = str(i)+"_test_dec_py_at01"
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
            #all_ensembles = ensembles['ensembles'][0]
            for cont_ensemble in all_ensembles:
                #print(cont,sum(cont_ensemble['selected_channels'][0][0][0]))
                
                if(TYPE):
                    selected_channels = ensemble_sett['channels'][cont]
                    mean = ensemble_sett['means'][cont][0]
                    std  = ensemble_sett['means'][cont][1]
                    clf = cont_ensemble
                else:
                    index_best_perf = np.argmax(cont_ensemble['all_perfs'][0][0][0])
                    #selected_channels = cont_ensemble['selected_channels'][0][0][0]
                    selected_channels = cont_ensemble['all_keep_chan'][0][0][index_best_perf]
                    mean = cont_ensemble['mean'][0][0][0]
                    std = cont_ensemble['std'][0][0][0]
                    clf = joblib.load(cont_ensemble['clf_filename'][0][0][0])
                cont = cont+1
                
                test_base = bases.create_base(samples,selected_channels)
                norm_test = m2.normalization(test_base,mean,std)
                norm_test[:,-1] = test_base[:,-1]
                
                inputs_test = norm_test[:,:-1]    
                targets_test = norm_test[:,-1]
                
                
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
        print(result_filename)
        print(conf.subject,perfs)
        
# 17 CLASSIFICADORES        
# 4 CANAIS PRE-FIXADOS ['PO7','Pz','Cz', 'PO8']
# [18, 35, 40, 50, 56, 67, 70, 71, 72, 79, 82, 83, 86, 92, 92] A at01
# [39, 59, 63, 72, 77, 83, 86, 86, 92, 93, 94, 95, 91, 94, 96] B at01 
# 4 CANAIS PRE-FIXADOS ['Pz','Oz', 'PO8','PO7']
# [15, 34, 54, 52, 64, 66, 76, 73, 75, 83, 87, 91, 91, 92, 95] A at01
# [41, 59, 72, 79, 81, 86, 87, 90, 92, 94, 93, 97, 94, 96, 97] B at01    
# 4 CANAIS PRE-FIXADOS ['Pz','Oz','O1', 'O2']
# [18, 31, 43, 53, 64, 67, 72, 75, 75, 83, 85, 88, 90, 92, 94] A at01
# [42, 52, 64, 73, 80, 84, 85, 86, 93, 94, 94, 95, 92, 95, 95] B at01
# 2 CANAIS PRE-FIXADOS ['PO7', 'PO8']
# [22, 31, 45, 56, 61, 63, 71, 72, 77, 82, 81, 88, 88, 89, 91] A at01
# [45, 60, 68, 74, 82, 88, 88, 90, 93, 94, 93, 95, 93, 95, 97] B at01
# 2 CANAIS PRE-FIXADOS ['PO7', 'PO8'] LDA
# [19, 34, 48, 56, 56, 69, 72, 76, 76, 84, 86, 89, 90, 95, 94] A at01 
# [41, 60, 63, 79, 81, 84, 88, 88, 91, 92, 91, 93, 91, 94, 95] B at01
# 64 CANAIS LDA
# [11, 8, 10, 14, 19, 16, 16, 21, 25, 25, 30, 32, 36, 32, 34] A at01 LDA
# [4, 12, 15, 15, 16, 19, 22, 26, 26, 34, 36, 39, 41, 38, 40] B at01 LDA
# ['C3', 'CP5', 'CP3', 'F7', 'TP7', 'Pz', 'PO7', 'POz', 'PO8', 'O1', 'Iz']
# [10, 18, 26, 32, 41, 42, 46, 47, 57, 64, 65, 61, 69, 73, 78] A at01
# [37, 46, 50, 63, 73, 79, 80, 83, 89, 89, 90, 89, 90, 90, 91] B at01
# ['Fz','Cz','Pz', 'Oz', 'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8']
# [12, 15, 25, 31, 34, 41, 43, 44, 52, 56, 56, 59, 63, 68, 71] A at01
# [30, 43, 57, 63, 73, 81, 83, 85, 87, 87, 88, 92, 94, 95, 95] B at01
# [12, 21, 30, 34, 37, 43, 46, 48, 49, 65, 65, 71, 71, 76, 78] A at01 LDA
# [35, 46, 52, 58, 65, 72, 75, 78, 80, 84, 85, 86, 84, 85, 87] B at01 LDA
# ['PO7','Pz','CPz','P7', 'FC1','Cz', 'PO8', 'FC5']
# [12, 15, 22, 28, 25, 33, 41, 43, 44, 46, 52, 57, 58, 65, 70] A at01
# [37, 45, 52, 59, 64, 70, 74, 77, 80, 83, 86, 86, 83, 83, 86] B at01
# [14, 20, 23, 34, 41, 42, 50, 53, 58, 67, 67, 67, 68, 74, 78] A at01 LDA
# [35, 51, 57, 68, 74, 81, 83, 87, 89, 89, 89, 92, 89, 92, 93] B at01 LDA
        
# 5 CLASSIFICADORES 
# 64 CANAIS
# [20, 30, 37, 40, 47, 57, 59, 71, 70, 77, 81, 86, 87, 89, 93] A at01
# [39, 55, 59, 65, 74, 82, 86, 87, 90, 90, 91, 93, 93, 95, 94] B at01
# [10, 20, 26, 28, 38, 40, 49, 56, 58, 65, 71, 72, 70, 77, 79] A at01 LDA
# [20, 27, 40, 49, 57, 65, 64, 71, 77, 79, 78, 82, 84, 86, 88] B at01 LDA
# ['Fz','Cz','Pz', 'Oz', 'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8']  
# [13, 17, 24, 22, 24, 25, 29, 36, 40, 46, 51, 51, 55, 56, 62] A at01 
# [33, 42, 58, 61, 68, 72, 79, 82, 80, 88, 85, 88, 87, 93, 93] B at01
#                           LDA
# [11, 19, 26, 30, 32, 36, 41, 52, 55, 58, 60, 65, 65, 72, 77] A at01 LDA
# [31, 36, 51, 60, 70, 73, 79, 79, 80, 85, 84, 87, 89, 90, 88] B at01 LDA
# ['PO7','Pz','CPz','P7', 'FC1','Cz', 'PO8', 'FC5']
# [7, 12, 12, 15, 20, 18, 21, 22, 21, 28, 30, 30, 29, 31, 35]  A at01
# [29, 36, 53, 58, 61, 64, 69, 73, 72, 79, 79, 83, 80, 81, 83] B at01
#                           LDA
# [9, 17, 20, 19, 27, 31, 31, 38, 41, 50, 52, 56, 56, 60, 66]  A at01 LDA
# [32, 37, 50, 52, 61, 66, 68, 69, 73, 78, 79, 83, 83, 85, 90] B at01 LDA
# 4 canais iniciais
# [16, 26, 35, 41, 50, 47, 51, 62, 69, 72, 75, 83, 80, 85, 87] A at01
# [35, 45, 59, 69, 72, 77, 81, 81, 90, 92, 92, 94, 96, 93, 93] B at01
# [ 9, 20, 28, 35, 47, 52, 56, 56, 69, 75, 76, 78, 82, 86, 90] A at01 LDA      
# [39, 52, 65, 71, 73, 79, 84, 85, 90, 92, 93, 92, 92, 91, 94] B at01 LDA

        
# 1 CLASSIFICADOR
# 64 CANAIS
# [11, 13, 18, 22, 29, 32, 39, 47, 46, 52, 53, 59, 57, 64, 66] A at01
# [25, 31, 44, 55, 57, 62, 68, 68, 73, 75, 77, 79, 82, 81, 82] B at01
# CANAIS FIXOS ['Fz','Cz','Pz', 'Oz', 'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8']
# ###############[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] A at01
# [21, 33, 39, 41, 45, 54, 60, 66, 70, 73, 73, 76, 76, 77, 80] B at01
# CANAIS FIXOS ['PO7','Pz','CPz','P7', 'FC1','Cz', 'PO8', 'FC5']
# ###############[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] A at01        
# [24, 24, 34, 43, 49, 50, 59, 63, 61, 65, 64, 68, 69, 72, 71] B at01
# 4 canais iniciais
# [10, 13, 17, 19, 24, 26, 29, 36, 40, 43, 45, 54, 59, 62, 64] A at01
# [29, 41, 57, 63, 66, 72, 72, 79, 85, 85, 84, 87, 87, 88, 90] B at01
#                           LDA
# 64 CANAIS
# [12, 15, 25, 28, 31, 42, 51, 53, 55, 60, 58, 60, 68, 71, 73] A at01 LDA
# [28, 39, 46, 53, 61, 66, 71, 74, 78, 80, 80, 83, 84, 85, 86] B at01 LDA
# CANAIS FIXOS ['Fz','Cz','Pz', 'Oz', 'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8']
# [ 4,  7,  7, 10,  8, 10, 12, 17, 19, 19, 21, 24, 24, 26, 25] A at01 LDA
# [23, 30, 35, 44, 47, 55, 59, 66, 67, 71, 74, 74, 78, 80, 79] B at01 LDA
# CANAIS FIXOS ['PO7','Pz','CPz','P7', 'FC1','Cz', 'PO8', 'FC5']
# [ 4,  6, 10, 12, 14, 14, 16, 22, 23, 27, 32, 34, 40, 43, 44] A at01 LDA
# [23, 25, 39, 41, 47, 54, 59, 64, 68, 72, 73, 76, 78, 78, 78] B at01 LDA
# 4 canais iniciais
# [09, 18, 25, 24, 28, 33, 37, 45, 47, 50, 49, 55, 58, 62, 65] A at01 LDA
# [28, 38, 49, 57, 65, 69, 75, 75, 77, 80, 81, 81, 81, 81, 84] B at01 LDA
        
        
#ensembles = load_ensemble(conf,"07_26_15_05_25_train_dec_py_at01")# 97%
#ensembles = load_ensemble(conf,"07_26_21_48_01_train_dec_py_at01")# 94%

# Seleção de canais 
# [18, 28, 49, 53, 65, 66, 74, 79, 80, 84, 88, 87, 89, 92, 92] Indivíduo A
# [42, 63, 62, 69, 81, 86, 90, 91, 94, 94, 94, 97, 95, 96, 97] Indivíduo B
        
        
# cs_07_29_12_00_34_train_dec_py_at01
# A [18, 28, 49, 53, 65, 66, 74, 79, 80, 84, 88, 87, 89, 92, 92] 
# cs_08_03_13_42_59_train_dec_py_
# A [14, 28, 42, 49, 60, 67, 71, 77, 79, 80, 86, 90, 88, 91, 92]
# cs4f_08_09_01_13_54_train_dec_py_at01
# A [18, 35, 40, 50, 56, 67, 70, 71, 72, 79, 82, 83, 86, 92, 92]
# cs4f_08_11_17_14_41_train_dec_py_at01
# A [18, 31, 43, 53, 64, 67, 72, 75, 75, 83, 85, 88, 90, 92, 94]
# cs4f_08_12_15_56_28_train_dec_py_at01
# A [22, 31, 45, 56, 61, 63, 71, 72, 77, 82, 81, 88, 88, 89, 91]
# cs4f_08_10_00_33_06_train_dec_py_at01
# A [15, 34, 54, 52, 64, 66, 76, 73, 75, 83, 87, 91, 91, 92, 95]
