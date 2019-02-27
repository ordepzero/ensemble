# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 01:03:04 2018

@author: PeDeNRiQue
"""

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
    

def load_samples(base,size_part=14):
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
  
selected_channels = [1]*64
nbOfChars = 1

conf = configuration.Configuration()   
conf.auto()   

ensembles = load_ensemble(conf,"only_B_")

for i in range(nbOfChars):
    prefix = "test_ss_"+str(i)+"_"
    filename = config_e.create_sample_mat_file(conf,prefix)
    #print(filename)
    
    signal_file_test = sio.loadmat(filename)


    samples = load_samples(signal_file_test,160)


test_base = bases.create_base(samples,selected_channels)

cont = 0
final_results = []
final_erros = []
c_values = []
for cont_ensemble in ensembles['ensembles'][0]:
    print(cont,sum(cont_ensemble['selected_channels'][0][0][0]))
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
for a,b in zip(targets_test,result):
    if(b == 1):
        c[int(a)-1] = c[int(a)-1]+1
        print(a,b)
    

 
    
    
    
    
    
    
    