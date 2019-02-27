# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 20:14:38 2018

@author: PeDeNRiQue
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 20:03:28 2018

@author: PeDeNRiQue
"""

import scipy.io as sio 
from os import listdir
from os.path import isfile, join

import configuration
import config_enviroment as ce


def process(temp_rs,base_index=0):
    
    perf = [0]*len(temp_rs[0]['test_perf'][0])
    conf = [[]]*len(temp_rs[0]['test_perf'][0])
    
    for trs in range(len(temp_rs)):
        for tr in range(len(temp_rs[trs]['test_perf'][0])):
            value = temp_rs[trs]['test_perf'][0][tr]
            print(trs,tr,len(perf))
            if(value > perf[tr]):
                perf[tr] = value
                conf[tr] = temp_rs[trs]
                '''
                conf[tr] = {'test_base_index':tr+base_index,
                            'algoritmo': temp_rs[trs]['algoritmo'][tr],
                           'kernel': temp_rs[trs]['kernel'][tr],
                           'C': temp_rs[trs]['C'][0][tr], 
                           'coef0': temp_rs[trs]['coef0'][0][tr], 
                           'degree': temp_rs[trs]['degree'][0][tr], 
                           'gamma': temp_rs[trs]['gamma'][0][tr],
                           'subject': temp_rs[trs]['subject'][tr],
                           'lowcut': temp_rs[trs]['lowcut'][0][tr],
                           'highcut': temp_rs[trs]['highcut'][0][tr],
                           'filter_order': temp_rs[trs]['filter_order'][0][tr],
                           'test_acc':temp_rs[trs]['test_acc'][0][tr],
                           'test_perf':temp_rs[trs]['test_perf'][0][tr]}
                '''

    return conf,perf

def convert_ens_conf_mat(ens_conf):
    
    ens_conf_dic = {'test_base_index':[],
            'algoritmo': [],
           'kernel': [],
           'C': [], 
           'coef0': [], 
           'degree': [], 
           'gamma': [],
           'subject': [],
           'lowcut': [],
           'highcut': [],
           'filter_order': [],
           'test_acc':[],
           'test_perf':[],
           'mean':[],
           'std':[]}
    
    for t_ens_conf in ens_conf:
        ens_conf_dic['test_base_index'].append(t_ens_conf['test_base_index'])
        ens_conf_dic['algoritmo'].append(t_ens_conf['algoritmo'])
        ens_conf_dic['kernel'].append(t_ens_conf['kernel'])
        ens_conf_dic['C'].append(t_ens_conf['C'])
        ens_conf_dic['coef0'].append(t_ens_conf['coef0'])
        ens_conf_dic['degree'].append(t_ens_conf['degree'])
        ens_conf_dic['gamma'].append(t_ens_conf['gamma'])
        ens_conf_dic['subject'].append(t_ens_conf['subject'])
        ens_conf_dic['lowcut'].append(t_ens_conf['lowcut'])
        ens_conf_dic['highcut'].append(t_ens_conf['highcut'])
        ens_conf_dic['filter_order'].append(t_ens_conf['filter_order'])
        ens_conf_dic['test_acc'].append(t_ens_conf['test_acc'])
        ens_conf_dic['test_perf'].append(t_ens_conf['test_perf'])
        ens_conf_dic['mean'].append(t_ens_conf['mean'])
        ens_conf_dic['std'].append(t_ens_conf['std'])
    
    return ens_conf_dic

if __name__ == "__main__": 
    conf = configuration.Configuration()   
    conf.auto()
    conf.subject = "A"
    conf.filter_ordem = 5
    conf.highcut = 10
    result_directory = ce.DIRECTORY_BASE+"/results_"+conf.subject+"/"
    
    #subdirectory = "fc_5b08_25_02_03_49_train_dec_py_at01"
    subdirectory = "cs4f_1b_svm_09_07_09_37_35_train_dec_py_at01"
    #subdirectory = "fc_5b08_25_02_05_11_train_dec_py_at01"
    #subdirectory = "fc_5b08_25_02_03_49_train_dec_py_at01"
    result_directory = result_directory+subdirectory
    
    onlyfiles = [f for f in listdir(result_directory) if (isfile(join(result_directory, f)) and f.endswith(".mat"))]
    rs1 = []
    rs2 = []
    for r in onlyfiles:
        rs1.append(sio.loadmat(result_directory+"/"+r))
       
    perfs = [0]*1
    results_configs = [[]]*1

    for cont_rs1 in range(len(rs1)):
        temp_index = rs1[cont_rs1]['test_base_index'][0][0]
        #print(rs1[cont_rs1]['test_base_index'][0][0])
        
        if(rs1[cont_rs1]['subject'][0] == conf.subject):
            if(rs1[cont_rs1]['filter_order'][0][0] == conf.filter_ordem):
                if(rs1[cont_rs1]['highcut'][0][0] == conf.highcut):
                    if(rs1[cont_rs1]['test_perf'][0][0] >= perfs[temp_index]):
                        perfs[temp_index] = rs1[cont_rs1]['test_perf'][0][0]
                        results_configs[temp_index] = rs1[cont_rs1] 

    
    result_file = ce.create_ensemble_file(conf,subdirectory)
    sio.savemat(result_file, {'ensembles':results_configs})

    
    if(False):
        for cont_rs1 in rs1:
            print(cont_rs1["C"][0][0],cont_rs1["coef0"][0][0],cont_rs1["degree"][0][0],cont_rs1["gamma"][0][0],cont_rs1['test_perf'][0][0])
    else:
        for cont_rs1 in rs1:
            print(cont_rs1['test_perf'][0][0])











