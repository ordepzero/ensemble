# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:01:22 2018

@author: PeDeNRiQue
"""

import random
import copy
import numpy as np

import configuration as config
import config_enviroment as config_e
import main as core
import bases
import svm as svm
import linear_discriminant_analysis as lda

def create_bases_with_new_conf():
    conf = config.Configuration()   
    conf.auto()
    
    for s in ["A","B"]:
        for i in [10.0,20.0]:
            for j in [5,8]:
        
                conf.filter_ordem = j
                conf.highcut = i
                conf.subject = s
                            
                core.load_samples2(conf)

def create_classifiers(n_classifiers,classifier,params=None):
    
    classifiers = []
    if(classifier == "lda"):
        for i in range(n_classifiers):
            temp_classifier = lda.LDA()
            classifiers.append(temp_classifier)
    elif(classifier == "svm"):
        if(params == None):
            for i in range(n_classifiers):
                temp_classifier = svm.SVM("linear")
                classifiers.append(temp_classifier)
        else:
            for i in range(n_classifiers):
                temp_classifier = svm.SVM("poly",params)
                classifiers.append(temp_classifier)
            
            
    return classifiers

    
def avaliate_selected_channels(base_of_samples,selected_channels,index_base,conf):

    if(index_base < 8):        
        base_train,base_test = bases.create_train_test_bases(base_of_samples[0:8],selected_channels,index_base)
    else:
        base_train,base_test = bases.create_train_test_bases(base_of_samples[8:17],selected_channels,index_base-8)

    #print(len(base_train),len(base_test))              
    classifier = svm.SVM("poly",conf.params)            
    classifier.execute_training(base_train)
    results = classifier.execute_test(base_test)  
    
    return results[3]

def avaliate_selected_channels2(base_of_samples,selected_channels,index_base):
    if(index_base < 8):        
        base_train,base_test = bases.create_train_test_bases(base_of_samples[0:8],selected_channels,index_base)
    else:
        base_train,base_test = bases.create_train_test_bases(base_of_samples[8:17],selected_channels,index_base-8)

    total = 0
    cont = 0
    for i in range(len(base_test)):
        temp_total = 0
        for j in range(len(base_test[i])):
            cont = cont + 1
            temp_total = temp_total + base_test[i][j]
        total = total + temp_total
    return total/cont#,base_train,base_test



def select_channels(base_of_samples,index_base,conf,config_e,result_filename):
    
    fo = open(result_filename,"w+") 
    fo.write(conf.get_info())
                
    selected_channels = [1]*64
    
    
    result = avaliate_selected_channels(base_of_samples,selected_channels,index_base)
    
    best_results = []
    best_channels = []
    best_results.append(result)
    best_channels.append(selected_channels)
    
    
    while(True):
        print(sum(selected_channels))
        if(sum(selected_channels) == 10 or len(best_results) == 64):
            break
        
        temp_result = None
        temp_selected_channels = None
        
        for i in range(len(selected_channels)):
            
            if(selected_channels[i] == 1):
                selected_channels[i] = 0        
                result = avaliate_selected_channels(base_of_samples,selected_channels,index_base,conf)
                
                if(temp_result == None):
                    temp_result = result
                    temp_selected_channels = copy.copy(selected_channels)
                else:
                    if(result > temp_result):
                        temp_result = result
                        temp_selected_channels = copy.copy(selected_channels)

                selected_channels[i] = 1
               
        #print(temp_selected_channels,temp_result)
        best_results.append(temp_result)
        best_channels.append(temp_selected_channels)
        selected_channels = copy.copy(temp_selected_channels)
    print(temp_result)
        
    return best_results,best_channels

def select_channels2(base_of_samples,index_base,conf,config_e,result_filename):
    
    fo = open(result_filename,"w+") 
    fo.write(conf.get_info())
                
    selected_channels = [0]*64
    
    
    #result = avaliate_selected_channels(base_of_samples,selected_channels,index_base)
    
    best_results = []
    best_channels = []
    #best_results.append(result)
    #best_channels.append(selected_channels)
    
    
    while(True):
        print(sum(selected_channels))
        if(sum(selected_channels) == 32 or len(best_results) == 64):
            break
        
        temp_result = None
        temp_selected_channels = None
        
        for i in range(len(selected_channels)):
            
            if(selected_channels[i] == 0):
                selected_channels[i] = 1        
                result = avaliate_selected_channels(base_of_samples,selected_channels,index_base)
                
                if(temp_result == None):
                    temp_result = result
                    temp_selected_channels = copy.copy(selected_channels)
                else:
                    if(result > temp_result):
                        temp_result = result
                        temp_selected_channels = copy.copy(selected_channels)

                selected_channels[i] = 0
               
        #print(temp_selected_channels,temp_result)
        best_results.append(temp_result)
        best_channels.append(temp_selected_channels)
        selected_channels = copy.copy(temp_selected_channels)
    print(temp_result)
        
    return best_results,best_channels

def select_channels3(base_of_samples,index_base,conf,config_e,result_filename):
    cont = 0
    fo = open(result_filename,"w+") 
    fo.write(conf.get_info())
    
    
    best_results = []
    best_channels = []
    
    selected_channels = [0] * 64
    temp_selected_channels = []

    while(sum(selected_channels) < 40 and cont < 10):
        cont = cont + 1
        temp_vector = None
        temp_avaliate = None
        
        for k in range(15):
            
            available_channels = []
            temp_selected_channels = []
            
            for i in range(len(selected_channels)):
                if(selected_channels[i] == 0):
                    available_channels.append(i)
            
            for i in range(4):
                temp_item_index = random.choice(range(len(available_channels)))
                temp_selected_channels.append(available_channels[temp_item_index])
                temp_item_value = available_channels.pop(temp_item_index)
            
            for i in range(len(temp_selected_channels)):
                selected_channels[temp_selected_channels[i]] = 1
                
            result = avaliate_selected_channels(base_of_samples,selected_channels,index_base,conf)
            
          
            if(temp_vector == None):    
                temp_vector = copy.copy(selected_channels)
                temp_avaliate = result
            elif(result > temp_avaliate):
                temp_vector = copy.copy(selected_channels)
                temp_avaliate = result     
            
            channels_s = ""    
            for i in range(len(temp_vector)):
                if(temp_vector[i] == 1):
                    channels_s = channels_s + str(i)+" "
            print(result,sum(selected_channels),"-",channels_s)
            
            for i in range(len(temp_selected_channels)):
                selected_channels[temp_selected_channels[i]] = 0
            
          
        if(len(best_results) == 0 or temp_avaliate > best_results[-1]):
            selected_channels = copy.copy(temp_vector)      
            best_results.append(temp_avaliate)
            best_channels.append(copy.copy(selected_channels))
        
    print(temp_avaliate)
        
    return best_results,best_channels

if __name__ == "__main__":
    
    #create_bases_with_new_conf()
    
    if(True):
    
        classifier = "svm"
        n_partitions = 17
        n_signals_cycle = 12
        n_cycles_char = 15
        n_signals_char = n_signals_cycle * n_cycles_char
        n_char_partition = 5
        
        
        config_e.create_directories()    
        
        conf = config.Configuration()   
        conf.auto()
        
        result_filename = config_e.create_result_file(conf)
                    
        samples = core.load_samples2(conf)
        
        selected_channels = [1]*64
        #selected_channels[60] = 1
        
        if(False):
            sct = ['Fz', 'Cz', 'Pz', 'Oz', 'C3', 'C4', 'P3', 'P4', 'PO7', 'PO8']
            
            for i in range(len(selected_channels)):
                if(core.channels_names[i] in sct):
                    selected_channels[i] = 1
            
        
        bases_of_samples = bases.create_sequential_bases(samples,n_partitions)
        
        #classifiers = create_classifiers(n_partitions,classifier,conf.params)
        classifiers = create_classifiers(n_partitions,classifier)
        trainning_bases = []
        code_bases = []
        
        line_column_values = []
        line_column_results = []
        selected_channels_partition = []
        all_matrix = []
        
        
        #for i in range(1):
        #for i in range(len(bases_of_samples)):
        #    esp_selected_channels = select_channels3(bases_of_samples,i,conf,config_e,result_filename)
            
        #    selected_channels_partition.append(esp_selected_channels[1][-1])
        
        if(False):
            selected_channels_file = open("selected_channels_2.txt","w")            
            
            for i in selected_channels_partition:
                selected_channels_file.write(str(i))
                selected_channels_file.write("\n")
            selected_channels_file.close()
            
            
        if(len(selected_channels_partition) == 0):
            for i in range(len(bases_of_samples)): 
                selected_channels_partition.append(selected_channels)
        
        
        # line_column_values has the value of intensified line/column 
        for i in range(len(bases_of_samples)):
            for j in range(0,len(bases_of_samples[i]),n_signals_char):
                temp_codes = bases_of_samples[i][j:j+n_signals_cycle]
                temp_line_colunm = []
                for k in range(len(temp_codes)):
                    if(temp_codes[k].target == 1):
                        temp_line_colunm.append(int(temp_codes[k].code))
                line_column_values.append(temp_line_colunm)
        
        # create training bases and the ordem of intensifications of lines/columns
        for i in range(len(bases_of_samples)):
            temp_train_base = bases.create_base(bases_of_samples[i],selected_channels_partition[i])
            trainning_bases.append(temp_train_base)
            temp_code_base = bases.create_code_base(bases_of_samples[i])
            code_bases.append(temp_code_base)
            
        for i in range(len(classifiers)):
            classifiers[i].execute_training(trainning_bases[i])
            
        for i in range(n_partitions):
            first_partition_index = n_signals_char*n_char_partition*i
            
            for j in range(n_char_partition):        
                
                result = 0
                sequence_begin = n_signals_char * j
                sequence_end = sequence_begin + n_signals_char
                
                char_signals = trainning_bases[i][sequence_begin:sequence_end]
                codes_values = code_bases[i][sequence_begin:sequence_end]              
                
                SCL = np.array([[0] * 6] * 6)
                
                for k in range(len(char_signals)):
                    line_column = int(codes_values[k])
                    
                    for l in range(n_partitions):
                    
                        if(i != l):                                                                                                                                                                                                                                                                                                                                                                                                                   
                            #result = result + classifiers[l].execute_test(np.array([char_signals[k]]))
                            result = result + classifiers[l].decision_function(np.array(char_signals[k]))
                    if(line_column < 7):
                        for m in range(6):
                            SCL[m][line_column-1] = SCL[m][line_column-1] + result
                    else:
                        for m in range(6):
                            SCL[line_column-7][m] = SCL[line_column-7][m] + result
                   
                l = 0
                c = 0
                for i in range(len(SCL)):
                    for j in range(len(SCL[i])):
                        if(SCL[i][j] > SCL[l][c]):
                            l = i
                            c = j
                l = l + 7
                c = c + 1
                temp_values = []
                temp_values.append(c)
                temp_values.append(l)
                line_column_results.append(temp_values)
                all_matrix.append(copy.copy(SCL))
                #print(SCL)
               
        for i in range(len(line_column_values)):
            line_column_values[i].sort()
            print(line_column_values[i],line_column_results[i]) 
            print(all_matrix[i])
    
    
def teste1():

    config_e.create_directories()    
    
    conf = config.Configuration()   
    conf.auto()
    
    result_file = config_e.create_result_file(conf)
                
    samples = core.load_samples2(conf)
    
    selected_channels = [1]*64
    #selected_channels[60] = 1
    #sct = ['C3', 'CP5', 'CP3', 'F7', 'TP7', 'Pz', 'PO7', 'POz', 'PO8', 'O1', 'Iz']
    
    base_of_samples = bases.create_sequential_bases(samples,17)

    classifiers = create_classifiers(17)
    trainning_bases = []
    
    for i in range(len(base_of_samples)):
        temp_train_base = bases.create_base(base_of_samples[i],selected_channels)
        trainning_bases.append(temp_train_base)
        
    for i in range(len(classifiers)):
        classifiers[i].train(trainning_bases[i])
        
    which_base = 0
    which_char = 4
    char_index = 180*which_char
    n_signals = 180
    char1 = np.array(trainning_bases[which_base][char_index:char_index+n_signals])
        
    SCL = np.array([[0] * 6] * 6)
    CONSTANT_M = 1 #(1/180)*(1/17)
    
    global_index = (which_base * 900) + char_index
            
    for i in range(len(char1)):
        line_column = int(samples[global_index+i].code)
        
        result = 0
        
        for j in range(len(classifiers)):   
            
            result = result+classifiers[j].my_lda.predict_proba([char1[i][:-1]])
            
            #print(result)
        final_result = CONSTANT_M * result[0][1]
        
        #print(i,j,final_result,result)
        if(line_column < 7):
            for k in range(6):
                SCL[k][line_column-1] = SCL[k][line_column-1] + final_result
        else:
            for k in range(6):
                SCL[line_column-7][k] = SCL[line_column-7][k] + final_result
        
        
    print(SCL)
    for i in samples[global_index:global_index+12]:
        if(i.target == 1):
            print(i.code)    
        
'''
WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU

MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR
'''