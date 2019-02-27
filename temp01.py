# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:41:41 2018

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
import step_03_part_02 as step3
import step_01_part_02 as step1

testA='WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU';
testB='MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR';

matrix=['ABCDEF','GHIJKL','MNOPQR','STUVWX','YZ1234','56789_']

def test():
#if __name__ == "__main__": 
    
    conf = configuration.Configuration()   
    conf.auto() 
    conf.subject="A"
    conf.filter_ordem=5
    conf.highcut=10 
    
    minimun_avaliation = str(0.30).replace(".","")

    result_filename = "cs4f_08_10_00_33_06_train_dec_py_at01"
    result_directory = config_e.DIRECTORY_BASE+"/results_"+conf.subject+"/"+result_filename
    new_ensemble = sio.loadmat(result_directory+"/"+minimun_avaliation+".mat")
    
    ensembles = []
    for cont_ensemble in range(17):
        temp_ensemble = joblib.load(result_directory+"/"+str(cont_ensemble)+"_"+minimun_avaliation+".pkl")
        ensembles.append(temp_ensemble)
    

#def temp():
if __name__ == "__main__": 
    minimun_avaliation = str(0.25).replace(".","")
    nbOfChars = 85
    n_bases = 17
    conf = configuration.Configuration()   
    conf.auto() 
    conf.subject="B"
    conf.filter_ordem=5
    conf.highcut=10 
    n_trials_to_test = 15
    list_n_trials_to_test = range(1,16)
    prefix = "_train_dec_py_at01"
    
    samples = step1.load_all_samples(config_e,conf,nbOfChars,prefix)
    
    result_filename = "cs4f_08_10_00_33_17_train_dec_py_at01"
    result_directory = config_e.DIRECTORY_BASE+"/results_"+conf.subject+"/"+result_filename
    new_ensemble = sio.loadmat(result_directory+"/"+minimun_avaliation+".mat")
    
    bases_of_samples = bases.create_sequential_bases(samples,n_bases)
    means_stds = []
    classifiers = []
    selected_channels = [1]*64
    cont_n_bases = 0
    for cont_bases_of_samples,cont_new_ensemble in zip(bases_of_samples,new_ensemble['channels']):
        temp_base = bases.create_base(cont_bases_of_samples,cont_new_ensemble)
        norm,mean,std = step1.normalization(temp_base)
        norm[:,-1] = temp_base[:,-1]
        means_stds.append([mean,std])
        
            
        params = {'C':1,
                  'coef0':0,
                  'gamma':'auto',
                  'degree':3}
        
        clf = step1.create_svm("linear",params,norm)
        joblib.dump(clf, result_directory+"/"+str(cont_n_bases)+"_"+minimun_avaliation+".pkl")
        cont_n_bases = cont_n_bases + 1
       
    new_ensemble = {'perfs':new_ensemble['perfs'][0],
                    'channels':new_ensemble['channels'],
                    'means':means_stds}
    
    sio.savemat(result_directory+"/"+minimun_avaliation+".mat", new_ensemble)
