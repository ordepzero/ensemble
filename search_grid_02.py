# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 04:31:00 2018

@author: PeDeNRiQue
"""


import os
import datetime
from sklearn import svm


import configuration as config
import config_enviroment as config_e
import sample
import bases
import step_01

#Search Grid
import pandas as pd 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pickle

#Listar os arquivos
from os import listdir
from os.path import isfile, join

#O numero no nome das bases indica o número de bases utilizadas vezes  1020
RES_DIRECTORY = "res"
PREFIX = ""
TOTAL_PERF = RES_DIRECTORY+"/"+RES_DIRECTORY+"_totalPerformances_3_1.pkl"
BEST_ESTIMATOR = RES_DIRECTORY+"/"+RES_DIRECTORY+"_bestEstimator_3_1.pkl"
BEST_PARAMS = RES_DIRECTORY+"/"+RES_DIRECTORY+"_bestParams_3_1.pkl"

def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

    
def separate_att_labels(samples,selected_channels):
    dataX = []
    
    #print(selected_channels)
    for i in range(len(samples)):
        dataX.append(samples[i].get_signals(selected_channels))
    
    #Normalização
    #scaler = MinMaxScaler()
    #scaler.fit(dataX)
    #dataX = scaler.transform(dataX)
    
    dataX,mean,std = step_01.normalization(dataX)
    
    dataY = []
    
    for i in range(len(samples)):
        dataY.append(samples[i].target)
        
    return dataX,dataY

    
def execute_searchgrid(dataX,dataY,filename):    
    
    print("Tamanho da base",len(dataX))
    #cs = [1,10,100]
    #gamma = [0.01,0.1,1,10]
    #coef0 = [0,1]
    #degree = [2,3,4,5]
    
    #cs = [0.01,0.05,0.1,0.5,1.0]
    #gamma = [0.1,0.5,1]
    #coef0 = [0,1]
    #degree = [1,2,3]
    
    
    #Experimento 2: com valores dos parâmetros maiores
    #cs = [0.01,0.05,0.1,0.5,1.0]#,2.0,5.0,10.0]
    cs = [0.01,0.1,0.5,1,10]#Experimento 3
    gamma = [0.01,0.1,0.5,1,10,1e2]
    coef0 = [1e-5,1,1e3]
    degree = [3]
    
    params = {'C':cs,'coef0':coef0,'degree':degree,'gamma':gamma}
    grid = GridSearchCV(svm.SVC(kernel="poly"),params,cv=2,verbose=3,n_jobs=1,return_train_score=True)
    
    grid.fit(dataX,dataY)
    
    if not os.path.exists(RES_DIRECTORY):
        os.makedirs(RES_DIRECTORY)
    
    with open(TOTAL_PERF,"wb") as f:
    	pickle.dump(grid.cv_results_,f,pickle.HIGHEST_PROTOCOL)
    
    with open(BEST_ESTIMATOR,"wb") as f:
    	pickle.dump(grid.best_estimator_,f,pickle.HIGHEST_PROTOCOL)
    
    with open(BEST_PARAMS,"wb") as f:
    	pickle.dump(grid.best_params_,f,pickle.HIGHEST_PROTOCOL)
    
    with open(TOTAL_PERF,"rb") as f:
    	data = pickle.load(f)
    pd.DataFrame.from_dict(data).to_csv(filename+"final_perfs.csv",sep=",")
    
    return pd

#def execute():
if __name__ == "__main__":     
    
    if(True):
        
        t = str(datetime.datetime.now()) 
        subdic_name = t.replace("-","_").replace(":","_").replace(" ","_")[:19]
        prefix = "_train_dec_py_at01"
        n_all_char = 85
        
        filename = subdic_name[5:]+prefix
        
        config_e.create_directories()    
        
        conf = config.Configuration()   
        conf.auto()
        
        
        
        samples = step_01.load_all_samples(config_e,conf,n_all_char,prefix)
        
        #selected_channels = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
        selected_channels = [0]*64
        selected_channels[59] = 1
        
        #base_of_samples = bases.create_seq_balanced_bases(samples)
        
        samples = samples[0:180*5]
        
        dataX,dataY = separate_att_labels(samples,selected_channels)
        
        execute_searchgrid(dataX,dataY,filename)
        
        #execute() 
    else:
    #clf = svm.SVC(kernel='poly')
            
    #clf.fit(dataX,dataY)  
    
       
        mypath = RES_DIRECTORY
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
        all_pkls = []
        for f in onlyfiles:
            all_pkls.append(load_pkl(RES_DIRECTORY+"/"+f))
            
    
        for i in range(22,33):
            print(str(i)+" ",end="")
            print(max(all_pkls[i]['split1_test_score']))
            
        '''
        dict_keys(['mean_fit_time', 'std_fit_time', 
        'mean_score_time', 'std_score_time', 'param_C', 
        'param_coef0', 'param_degree', 'param_gamma', 
        'params', 'split0_test_score', 'split1_test_score', 
        'mean_test_score', 'std_test_score', 'rank_test_score', 
        'split0_train_score', 'split1_train_score', 
        'mean_train_score', 'std_train_score'])
        '''