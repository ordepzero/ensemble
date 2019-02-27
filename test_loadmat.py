# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:48:28 2018

@author: PeDeNRiQue
"""


import numpy as np
import scipy.io as sio 
from os import listdir
from os.path import isfile, join
from sklearn.externals import joblib
import configuration
import config_enviroment as config_e
import step_03


channels_names = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1",
                  "Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4",
                  "CP6","FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8","F7",
                  "F5","F3","F1","Fz","F2","F4","F6","F8","FT7","FT8",
                  "T7","T8","T9","T10","TP7","TP8","P7","P5","P3","P1",
                  "Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8",
                  "O1","Oz","O2","Iz"]

def get_channels_names(channels_selected):
    chann_names = []
    
    for cont_cs in range(len(channels_selected)):
        if(channels_selected[cont_cs] == 1):
            chann_names.append(channels_names[cont_cs])
    return chann_names


conf = configuration.Configuration()   
conf.auto() 
conf.subject="A"
conf.filter_ordem=5
conf.highcut=10 
n_trials_to_test = 15

B_files = ["07_26_15_05_25_train_dec_py_at01",
           "07_26_17_56_20_train_dec_py_at01",
           "07_26_18_03_35_train_dec_py_at01",
           "07_28_07_37_10_train_dec_py_at01",
           "cs_07_29_12_00_34_train_dec_py_at01",
           "cs_08_03_13_42_59_train_dec_py_",
           "cs4f_08_09_01_14_04_train_dec_py_at01",
           "cs4f_08_10_00_33_17_train_dec_py_at01",
           "fc_08_09_00_40_39_train_dec_py_at01",
           "fc_208_09_00_47_28_train_dec_py_at01",
           "fc_308_09_00_48_07_train_dec_py_at01",
           "cs4f_5b_08_25_20_11_36_train_dec_py_at01",
           "cs4f_1b_svm_09_07_09_37_23_train_dec_py_at01",
           "cs4f_lda_08_20_00_21_30_train_dec_py_at01",
           "cs4f_lda_5b08_28_00_38_19_train_dec_py_at01",
           "cs4f_1b_lda_09_06_22_21_17_train_dec_py_at01"]

A_files = ["07_26_21_48_01_train_dec_py_at01",
           "07_28_07_37_10_train_dec_py_at01",
           "07_28_07_19_49_train_dec_py_at01",
           "fc_lda_17b08_27_17_38_59_train_dec_py_at01",#
           "cs_07_29_12_00_34_train_dec_py_at01",#4 
           "cs_08_03_13_42_59_train_dec_py_", #5
           "cs4f_08_09_01_13_54_train_dec_py_at01", # 6 1 & 0.159
           "cs4f_08_12_15_56_28_train_dec_py_at01", # 7 1 & 0.0 & 0.0 & 0.0 & 0.120
           "cs4f_08_09_01_13_54_train_dec_py_at01", # 8 1 & 0.159
           "cs4f_08_11_17_14_41_train_dec_py_at01", # 9 1 & 0.144 # Usada no texto
           "cs4f_08_12_15_56_28_train_dec_py_at01", # 10 1 & 0.0 & 0.0 & 0.0 & 0.120 
           "cs4f_5b_08_25_20_11_28_train_dec_py_at01", #11 1 & 0.0 & 0.0 & 0.100
           "cs4f_1b_svm_09_07_09_37_35_train_dec_py_at01", 
           "cs4f_lda_08_20_00_21_03_train_dec_py_at01",
           "cs4f_lda_5b08_28_00_38_07_train_dec_py_at01",
           "cs4f_1b_lda_09_06_22_21_17_train_dec_py_at01"]

if(conf.subject == "A"):
    subdic_files = A_files 
else:
    subdic_files = B_files 

result_directory = config_e.DIRECTORY_BASE+"/results_"+conf.subject+"/"

#subdirectory = "07_28_07_37_10_train_dec_py_at01"
subdirectory = subdic_files[-4]
print(subdirectory)

result_directory = result_directory+subdirectory

onlyfiles = [f for f in listdir(result_directory) if (isfile(join(result_directory, f)) and f.endswith(".mat"))]

tf = sio.loadmat(result_directory+"/"+onlyfiles[0])

channels_frequenc = [0]*64
if(False):#Ordenar os canais mais selecionados
    for cont_only_files in onlyfiles:
        tf = sio.loadmat(result_directory+"/"+cont_only_files)
        ctf = joblib.load(tf['clf_filename'][0])
        #print(ctf.C,ctf.coef0,ctf.gamma)
        #print(tf['all_keep_chan'][1])
        channels_frequenc = np.add(channels_frequenc,tf['all_keep_chan'][-1])
        for cont_keep_chan in tf['all_keep_chan']:
            chann_names = get_channels_names(cont_keep_chan)
            #print(chann_names)
        
    t = zip(channels_frequenc,channels_names)
    t = sorted(t, key=lambda chann: chann[0])
    for cont_t in t:
        print(cont_t)
elif(True):
    for cont_only_files in onlyfiles:
        tf = sio.loadmat(result_directory+"/"+cont_only_files)
        #print(tf['all_perfs'][0])
        temp_cont_perfs = ""
        index_max_value = np.argmax(tf['all_perfs'][0])
        channels_frequenc = np.add(channels_frequenc,tf['all_keep_chan'][index_max_value])
        #print(index_max_value,tf['all_keep_chan'][index_max_value])
    t = zip(channels_frequenc,channels_names)
    t = sorted(t, key=lambda chann: chann[0],reverse=True)
    for cont_t in t:
        print(cont_t[0],cont_t[1])
            
elif(False):
    for cont_only_files in onlyfiles:
        tf = sio.loadmat(result_directory+"/"+cont_only_files)
        
        channels_frequenc = np.add(channels_frequenc,tf['selected_channels'][0])
        print(cont_only_files,sum(tf['selected_channels'][0]))
        
    t = zip(channels_frequenc,channels_names)
    t = sorted(t, key=lambda chann: chann[0],reverse=True)
    for cont_t in t:
        print(cont_t)
elif(False):#Mostrar todas as avaliações a medida que os canais eram selecionados
    if(False):
        for cont_only_files in onlyfiles:
            tf = sio.loadmat(result_directory+"/"+cont_only_files)
            #print(tf['all_perfs'][0])
            temp_cont_perfs = ""
            first_value = True
            for cont_perfs in tf['all_perfs'][0]:
                if(first_value):
                    first_value = False
                    temp_cont_perfs=str(cont_perfs)
                else:
                    temp_cont_perfs=temp_cont_perfs+";"+str(cont_perfs)
            print(temp_cont_perfs)
    if(True):#Imprimir a tabela em latex
        count = 0        
        
        for cont_only_files in onlyfiles:
            tf = sio.loadmat(result_directory+"/"+cont_only_files)
            #print(tf['all_perfs'][0])
            temp_cont_perfs = ""
            index_max_value = np.argmax(tf['all_perfs'][0])
            count = count + 1
            first_value = True
            for cont_perfs in range(len(tf['all_perfs'][0])): 
                current_value = tf['all_perfs'][0][cont_perfs]
                if(current_value == 0):
                    current_value = format(current_value, '.1f')
                else:
                    current_value = format(current_value, '.3f')
                if(first_value):
                    first_value = False
                    if(cont_perfs != index_max_value):
                        temp_cont_perfs=str(count)+" & "+str(current_value)
                    else:
                        temp_cont_perfs=str(count)+" & \cellcolor[HTML]{C0C0C0}"+str(current_value)
                else:
                    if(cont_perfs != index_max_value):
                        temp_cont_perfs=temp_cont_perfs+" & "+str(current_value)
                    else:
                        temp_cont_perfs=temp_cont_perfs+" & \cellcolor[HTML]{C0C0C0}"+str(current_value)
            temp_cont_perfs = temp_cont_perfs+" \\\ \hline"
            print(temp_cont_perfs)
elif(True):#Verificar quais canais foram mais selecionados
    '''
    for cont_cs in ['PO7','Pz','Cz', 'PO8']:
        cont = 0
        for cont_sn in channels_names:            
            if(cont_cs == cont_sn):
                print(cont_cs,cont)
                break
            else:
                cont = cont + 1
    '''
    ts = [0]*64
    for cont_only_files in onlyfiles:
        tf = sio.loadmat(result_directory+"/"+cont_only_files)
        ts = ts+tf['all_keep_chan'][-1]
    #print(ts)
    t = zip(ts,channels_names)
    t = sorted(t, key=lambda chann: chann[0], reverse=True)
    for cont_t in t:
        print(cont_t)
elif(False):
    for cont_only_files in onlyfiles:
        tf = sio.loadmat(result_directory+"/"+cont_only_files)
        print(tf['test_perf'][0][0])
elif(False):#SELECIONAR CANAIS DE ACORDO COM VALOR MINIMO
    
    minimun_avaliation = 0.25
    
    min_perfs = []
    min_chans = []
    sum_chans = 0
    for cont_only_files in onlyfiles:
        if(len(cont_only_files) < 10):
            continue
        tf = sio.loadmat(result_directory+"/"+cont_only_files)
        
        temp_index_min_ava = np.argmax(tf['all_perfs'][0])
        for cont_tf in range(len(tf['all_perfs'][0])):
            if(tf['all_perfs'][0][cont_tf] > minimun_avaliation):
                temp_index_min_ava = cont_tf
                break
        min_perfs.append(tf['all_perfs'][0][temp_index_min_ava])
        min_chans.append(tf['all_keep_chan'][temp_index_min_ava])
        sum_chans = sum_chans + temp_index_min_ava + 1
    print(sum_chans*4)
    print(min_perfs)
    
    new_ensemble = {"perfs":min_perfs,"channels":min_chans}
    #sio.loadmat(result_directory+"/"+onlyfiles[0])
    sio.savemat(result_directory+"/"+str(minimun_avaliation).replace(".","")+".mat", new_ensemble)
    #print(min_chans)
    
    
        