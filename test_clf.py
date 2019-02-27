# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:15:22 2018

@author: PeDeNRiQue
"""

from sklearn import svm
import scipy.io as sio 
import numpy as np

import configuration
import config_enviroment as config_e
import preprocessing as prep
import read_file as rf
import sample
import bases
import main_002 as m2

selected_channels = [1] * 64

conf = configuration.Configuration()   
conf.auto()

conf.subject = "B"
conf.highcut = 10
conf.filter_ordem = 5
test_base_index = 1

result_file = config_e.create_result_mat_file(conf)
filename = config_e.create_sample_mat_file(conf)

samples = m2.load_samples(filename)
bases_of_samples = bases.create_sequential_bases(samples,17)
    
bases_of_samples = bases_of_samples[0:8]

train,test = bases.create_train_test_bases(bases_of_samples,selected_channels,test_base_index)
        
norm_train,mean,std = m2.normalization(train)
norm_train[:,-1] = train[:,-1]

norm_test = m2.normalization(test,mean,std)
norm_test[:,-1] = test[:,-1]

classifier = svm.LinearSVC()

inputs = norm_train[:,:-1]
targets = norm_train[:,-1]

classifier.fit(inputs,targets)

#TESTAR

inputs_test = norm_test[:,:-1]
targets_test = norm_test[:,-1]
              
acc = classifier.score(inputs_test,targets_test)