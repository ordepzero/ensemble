# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:56:07 2017

@author: PeDeNRiQue
"""

# A função de decimação tem um filtro de ordem 8, testar colocar um
# parâmetro nessa função e mudar a ordem do filtro cheb

import os
import numpy as np
import random
import copy
from random import randint

import datetime
from scipy import signal
import matplotlib.pyplot as plt
import csv

import read_file as rf    
import neural_network as nn
import svm as svm
import linear_discriminant_analysis as lda
import filters
import configuration as config
import sample


channels_names = ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1",
                  "Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4",
                  "CP6","FP1","FPz","FP2","AF7","AF3","AFz","AF4","AF8","F7",
                  "F5","F3","F1","Fz","F2","F4","F6","F8","FT7","FT8",
                  "T7","T8","T9","T10","TP7","TP8","P7","P5","P3","P1",
                  "Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8",
                  "O1","Oz","O2","Iz"]
 
FIELDNAMES = ["Data", "Individuo","Algoritmo","Filtro","Ordem","Aten","Lowcut","Highcut","Size","DecTipo","Fator","Label","N","Normalizacao","PO8"]
DIRECTORY_BASE = "../../bases"
DIRECTORY_SIGNALS_CHANNELS = DIRECTORY_BASE+"/signals_channels"
DIRECTORY_SIGNALS_CHANNELS_A = DIRECTORY_SIGNALS_CHANNELS+"/signals_channels_A"
DIRECTORY_SIGNALS_CHANNELS_B = DIRECTORY_SIGNALS_CHANNELS+"/signals_channels_B"

def save_scv_file(time,config):
    with open(time, 'w+') as outcsv:
        WRITER = csv.DictWriter(outcsv, fieldnames=FIELDNAMES,delimiter=";")
        
        WRITER.writeheader()
        
        CSV_ROW = {FIELDNAMES[0]: time}
        CSV_ROW[FIELDNAMES[1]] =  config.subject
        CSV_ROW[FIELDNAMES[2]] =  config.alg
        CSV_ROW[FIELDNAMES[3]] =  config.filter_type
        CSV_ROW[FIELDNAMES[4]] =  config.filter_ordem
        CSV_ROW[FIELDNAMES[5]] =  config.aten        
        CSV_ROW[FIELDNAMES[6]] =  config.lowcut
        CSV_ROW[FIELDNAMES[7]] =  config.highcut
        CSV_ROW[FIELDNAMES[8]] =  config.size_part
        CSV_ROW[FIELDNAMES[9]] =  "cheby"
        CSV_ROW[FIELDNAMES[10]] =  config.downsampling
        CSV_ROW[FIELDNAMES[11]] =  "[0,1]"
        CSV_ROW[FIELDNAMES[12]] =  0
        CSV_ROW[FIELDNAMES[13]] =  "[-1,1]"
        
        #WRITER.writerow(row)
        #outcsv.close()
    return outcsv,WRITER,CSV_ROW

def save_config(config):   
    
    t = str(datetime.datetime.now()) 
    t = t.replace("-","_").replace(":","_").replace(" ","_")[:19]
    
    #outcsv,WRITER,CSV_ROW = save_scv_file(t,config)
#    subject = "B"
#    to_filter = True
#    to_decimate = False
#    filter_ordem = 3
#    downsampling = 10
#    lowcut = 0.05
#    highcut = 10.0
#    filter_type = "butter" #butter or cheby
    fo = open("tests_"+config.subject+"/"+t+"_fourier_01_11.txt", "w+")
    
    
    if(config.is_test):
        fo.write("TESTE\n")
    else:
        fo.write("DEFINITIVO\n")
    
    fo.write("Inicio: "+str(datetime.datetime.now())+"\n")
    fo.write("Individuo: "+config.subject+"\n")
    fo.write("Filtrado: "+str(config.to_filter)+"\n" )
    fo.write("Decimate: "+str(config.to_decimate)+"\n" )
    fo.write("Frequencia: "+str(config.freq)+"\n" )
    if(config.to_filter):
        fo.write("Filtro: "+config.filter_type+"\n")
        fo.write("Ordem: "+str(config.filter_ordem)+"\n")
        fo.write("Frequencia: "+str(config.lowcut)+" "+str(config.highcut)+"\n")
        if(config.filter_type == "cheby"):
            fo.write("AtenuacaoMinima: "+str(config.aten)+"\n")
    if(config.to_decimate):
        fo.write("Decimacao: "+str(config.downsampling)+"\n")
    fo.write("TamanhoDaParte: "+str(config.size_part)+"\n")
    fo.write("N Value: "+str(config.n_value)+"\n")
    
    if(config.alg=="svm"):
        fo.write("Algoritmo: SVM\n")
        fo.write("Kernel: Linear\n")
    elif(config.alg=="lda"):
        fo.write("Algoritmo: LDA\n")
    elif(config.alg=="rna"):
        fo.write("Algoritmo: RNA\n")
        
    return fo

def is_not_in_lines(lines, number):
    for i in range (len(lines)):
        if number == lines[i]:
            return 0
    return 1

    
def create_bases(base,n_bases = 5):
    
    base_of_ones = [] 
    base_of_zeros = [] 
    split_base = []
    zero_lines = []    
    one_lines = []    
    
    for x in base:
        if(x[-1] == 1):
            base_of_ones.append(x)
        else:
            base_of_zeros.append(x)
            
    base_of_ones = np.array(base_of_ones)
    base_of_zeros = np.array(base_of_zeros)
    
    n_samples = int(len(base_of_ones) / n_bases)
    
    bases = []   
    
    for j in range (n_bases):  
        split_base = []
        for i in range (n_samples):
            position = random.randint(0, len(base_of_ones)-1)
            while (not is_not_in_lines(zero_lines, position)):        
                position = random.randint(0, len(base_of_ones)-1)
            zero_lines.append(position)
            split_base.append(base_of_zeros[position])
            
            position = random.randint(0, len(base_of_ones)-1)
            while (not is_not_in_lines(one_lines, position)):        
                position = random.randint(0, len(base_of_ones)-1)
            one_lines.append(position)
            split_base.append(base_of_ones[position])
        
        bases.append(split_base)
        
    
    return np.array(bases)

def filter_decimation(signals,config):
    
    filtered_signals = []
    lowcut = config.lowcut
    highcut = config.highcut
    fe = config.freq
    order = config.filter_ordem
    aten = config.aten
    to_decimate = config.to_decimate
    
    for sig in signals:
        filtered_signals.append(filters.chebyshev1_filter(sig,lowcut,highcut,fe,order,aten,to_decimate,downsampling=config.downsampling))
        #filtered_signals.append(filters.butter_bandpass_filter(sig,lowcut,highcut,fe,order,to_decimate))
    return np.array(filtered_signals)
    
# ADICIONA O 'TARGET' ÀS PARTES DOS SINAIS
def load_base(specific_channel,config):
    
    if(config.to_decimate):
        specific_channel = filter_decimation(specific_channel,config)
    
    specific_channel = normalize_base(specific_channel)  
    return rf.create_base(specific_channel,config.subject)
    
def divide_base_and_execute(bases_of_samples,selected_channels,fo,alg="lda"):
    test = []
    train = []
    med_train_error = 0
    med_test_error = 0
    
    fo.write("Folds: "+str(len(bases_of_samples))+"\n")   
    fo.write("Canais: ")
    channels_s = ""
    for sc in range(len(selected_channels)):
        if(selected_channels[sc] == 1):
            channels_s = channels_s+ " " +channels_names[sc]
            fo.write(channels_names[sc]+" ")
            #fo.write(str(selected_channels[sc])+" ")
    fo.write("\n")    
    print(channels_s)
    for i in range(len(bases_of_samples)):
        test = []
        train = []
        for j in range(len(bases_of_samples)):
            if(j != i):
                for x in range(len(bases_of_samples[j])):
                    train.append(bases_of_samples[j][x].get_concat_signals(selected_channels))
            else:
                for x in range(len(bases_of_samples[j])):
                    test.append(bases_of_samples[j][x].get_concat_signals(selected_channels))
        train_error = 0
        test_error = 0
        
        train = np.array(train)
        test  = np.array(test)
        if(len(train) == 0 or len(test) == 0):
            break
        if(True):
            if(alg=="rna"):
                if(True):    
                    print("Executando RNA, Opcao 1")
                    train_error,test_error = nn.execute_rna(train,test,fo)
                elif(False):
                    print("Executando RNA, Opcao 2")
                    train_error,test_error = nn.execute_rna(train,test,fo,n_neuron=160,epoch=400)
            elif(alg=="lda"):
                #fo.write("Algoritmo: LDA\n")
                alg_lda = lda.LDA()
                train_error,test_error = alg_lda.execute_lda(train,test)
                mc = alg_lda.get_infos()
                print(mc)
            elif(alg=="svm"):
                alg_svm = svm.SVM()
                train_error,test_error = alg_svm.execute_svm(train,test)
                mc = alg_svm.get_infos()
                print(mc)
        
        fo.write("AcTreino:"+str(train_error)+"\tAcTeste:"+str(test_error)+"\n")
        for m in range(len(mc)):
            if(m < len(mc)-1):
                fo.write(str(mc[m])+" ")  
            else:
                fo.write(str(mc[m])+"\n")
        med_train_error += train_error
        med_test_error += test_error
        
        print(str(i)+" "+str(train_error)+" "+str(test_error))
        
    med_train_error = med_train_error/len(bases)
    med_test_error = med_test_error/len(bases)
    fo.write("AcMedioTreino: "+str(med_train_error)+"\tAcMedioTeste:"+str(med_test_error)+"\n")

    return med_test_error
    #CSV_ROW[FIELDNAMES[14]] =  med_test_error
        
    #WRITER.writerow(CSV_ROW)

def execute_for_all_channels(config):
    
    fo = save_config(config)
    
    for i in config.channels:
    #for i in [60,61,62]:
    #for i in range(55,64):
        print("Canal "+str(i)+" : "+channels_names[i])
        fo.write("Canal "+str(i+1)+" : "+channels_names[i]+"\n")
        signal_channel = rf.separate_signals([i],config.size_part,config.subject)
        
        
        base = load_base(signal_channel,config)
        bases = create_bases(base)
        
        divide_base_and_execute(bases,fo,config.alg)

    fo.write("Fim: "+str(datetime.datetime.now())+"\n")
    fo.close
    
def execute_configurations():
    
    list_subject = ["B","A"]    
    list_alg = ["lda"]    
    list_filter_type = ["cheby"] #butter or cheby or analog
    list_downsampling = [10]    
    list_to_filter = [True]
    list_to_decimate = [True,False]
    
    list_filter_orden = [1,3,5]
    list_lowcut = [0.5,5.0,10.0,15.0]
    list_highcut = [10.0,15.0,20.0]
    list_aten = [1.0]
    
    for sub in list_subject:
        for alg in list_alg:
            for ftype in list_filter_type:
                for samp in list_downsampling:
                    for filt in list_to_filter:
                        for dec in list_to_decimate:
                            for order in list_filter_orden:
                                for low in list_lowcut:
                                    for high in list_highcut:
                                        for aten in list_aten:
                                            print(sub,alg,ftype,samp,filt,dec,order,low,high,aten)
                                            if(low >= high):
                                                break
                                            
                                            conf = config.Configuration()
                                            conf.subject = sub                                           
                                            conf.alg = alg                                        
                                            conf.filter_type = ftype                                            
                                            conf.downsampling = samp
                                            conf.to_filter = filt
                                            conf.to_decimate = dec
                                            conf.filter_ordem = order
                                            conf.lowcut = low
                                            conf.highcut = high
                                            conf.aten = aten                                        
                                            conf.size_part = 160
                                            
                                            execute_for_all_channels(conf)
                                        
def find_into_bracket(params,is_number=False,has_bool = False):  
    
    t = params[params.find("[")+1:params.find("]")]
    if(is_number):
        numbers = []
        for n in t.split(","):
            numbers.append(float(n))
        return numbers
    if(has_bool):
        bools = []
        for n in t.split(","):
            bools.append(bool(n))
        return bools
    return t.split(",")
        

def load_execute_configurations():
    with open("experiments.txt") as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        
        
    for line in content:
        print(line)
        
        if("list_subject" in line):
            t = line.split("=")
            list_subject = find_into_bracket(t[1])
        elif("list_alg" in line):
            t = line.split("=")
            list_alg = find_into_bracket(t[1])
        elif("list_filter_type" in line):
            t = line.split("=")
            list_filter_type = find_into_bracket(t[1])
        elif("list_downsampling" in line):
            t = line.split("=")
            list_downsampling = find_into_bracket(t[1])
        elif("list_to_filter" in line):
            t = line.split("=")
            list_to_filter = find_into_bracket(t[1],has_bool=True)
        elif("list_to_decimate" in line):
            t = line.split("=")
            list_to_decimate = find_into_bracket(t[1],has_bool=True)
        elif("list_filter_orden" in line):
            t = line.split("=")
            list_filter_orden = find_into_bracket(t[1],is_number=True)
        elif("list_lowcut" in line):
            t = line.split("=")
            list_lowcut = find_into_bracket(t[1],is_number=True)
        elif("list_highcut" in line):
            t = line.split("=")
            list_highcut = find_into_bracket(t[1],is_number=True)
        elif("list_aten" in line):
            t = line.split("=")
            list_aten = find_into_bracket(t[1],is_number=True)
        elif("list_channels" in line):
            t = line.split("=")
            list_channels = find_into_bracket(t[1],is_number=True)
        elif("list_size_parts" in line):
            t = line.split("=")
            size_parts = find_into_bracket(t[1],is_number=True)
        elif("list_n_values" in line):
            t = line.split("=")
            n_values = find_into_bracket(t[1],is_number=True)
        elif("list_freq" in line):
            t = line.split("=")
            list_freq = find_into_bracket(t[1],is_number=True)
        elif("is_test" in line):
            t = line.split("=")
            is_test = find_into_bracket(t[1],has_bool=True)
            
    #print(len(range(int(list_channels[0]),int(list_channels[1]))))
    #return
            
    #print(list_subject,list_alg,list_filter_type,list_downsampling,list_to_filter,list_to_decimate,list_filter_orden,list_lowcut,list_highcut,list_aten)
    for sub in list_subject:
        for alg in list_alg:
            for ftype in list_filter_type:
                for samp in list_downsampling:
                    for filt in list_to_filter:
                        for dec in list_to_decimate:
                            for order in list_filter_orden:
                                for low in list_lowcut:
                                    for high in list_highcut:
                                        for aten in list_aten:
                                            for size_part in size_parts:
                                                for freq in list_freq:
                                                    print(sub,alg,ftype,samp,filt,dec,order,low,high,aten,size_part,freq)
                                                    if(low >= high):
                                                        break
                                                    
                                                    conf = config.Configuration()
                                                    conf.subject = sub                                           
                                                    conf.alg = alg                                        
                                                    conf.filter_type = ftype                                            
                                                    conf.downsampling = samp
                                                    conf.to_filter = filt
                                                    conf.to_decimate = dec
                                                    conf.filter_ordem = order
                                                    conf.lowcut = low
                                                    conf.highcut = high
                                                    conf.aten = aten                                        
                                                    conf.size_part = 160
                                                    conf.is_test = is_test
                                                    conf.channels = range(int(list_channels[0]),int(list_channels[1]))
                                                    conf.n_value = n_values[0]
                                                    conf.freq = freq
                                                    execute_for_all_channels(conf)

def normalize_base(base):
    new_base = []
    
    max_value = np.matrix(base).max()
    min_value = np.matrix(base).min()
    
    for line in base:
        values = []
        for value in line:
            if(True):            
                r = (value-min_value)/(max_value-min_value)
            elif(False):
                r = (2 *(value-min_value)/(max_value-min_value)) - 1
            values.append(r)
        new_base.append(values)
    
    return new_base
    
def main2():
    if(False):
        channels = [0]
        size_part = 160
        signal_channel = rf.separate_signals(channels,size_part)
        
        
        base = load_base(signal_channel)
        bases = create_bases(base)
        
        divide_base_and_execute(bases)
    elif(False):
        print("Inicio")
        conf = config.Configuration()
        execute_for_all_channels(conf)
    elif(False):
        print("Inicio")
        execute_configurations()
    elif(True):
        load_execute_configurations()
    elif(False):
        channels = [60]
        size_part = 160
        signal_channel = rf.separate_signals(channels,size_part,subject="B")
        
        conf = config.Configuration()
        conf.auto()

        base = load_base(signal_channel,conf)
        
        #if(True):
        #    base = normalize_base(base)
        
        bases = create_bases(base)
        
        '''
        s = filters.chebyshev1_filter(signal_channel[0],0.1,10,240,8,1,False,12)
        plt.plot(s)
        plt.show()
        287/305/291/308
        294/307
        
        '''
    else:   
        t = np.linspace(0, 160,14)
        channels = [0]
        size_part = 160
        signal_channel = rf.separate_signals(channels,size_part)
        base = load_base(signal_channel)
        
        
        #r = signal.decimate(base[0][0:160],14,axis=-1)
        #len(r)
    
        
        #plt.plot(base[0][0:160])
        plt.plot(base[0][0:160])
        plt.show()
        
def create_directories():
    
    if not os.path.exists(DIRECTORY_BASE):
        os.makedirs(DIRECTORY_BASE)
    
    if not os.path.exists(DIRECTORY_SIGNALS_CHANNELS):
        os.makedirs(DIRECTORY_SIGNALS_CHANNELS)
    
    if not os.path.exists(DIRECTORY_SIGNALS_CHANNELS_A):
        os.makedirs(DIRECTORY_SIGNALS_CHANNELS_A)
    
    if not os.path.exists(DIRECTORY_SIGNALS_CHANNELS_B):
        os.makedirs(DIRECTORY_SIGNALS_CHANNELS_B)
        

    dic_channels = [DIRECTORY_SIGNALS_CHANNELS_A,DIRECTORY_SIGNALS_CHANNELS_B]
    dic_files    = [rf.SIGNAL_FILE_A,rf.SIGNAL_FILE_B]
    
    print("Aguarde, isso pode levar um tempo")
    for dic_channel,dic_file in zip(dic_channels,dic_files):
        for channel in range(64):            
            file_to_save = dic_channel+"/signal_channel_"+str(channel)+".txt"
            if not os.path.exists(file_to_save):
                rf.read_specific_channels_and_save(dic_file," ",[channel],file_to_save);
    

def create_sequential_bases(samples,n_bases=5):
    bases = []  

    step = int(len(samples)/n_bases)
    
    if(step == 0):
        step = 1
    
    for i in range(0,len(samples),step):
        bases.append(samples[i:(i+step)])
        
    return bases
    
def create_bases_2(samples,n_bases=5):
    new_samples = []
    
    for i in range(len(samples)):
        if(samples[i].target == 1):
            new_samples.append(samples[i])
            begin = i - 5
            end = i + 5
            
            if(begin < 0):
                begin = 0
            if(end >= len(samples)):
                end = len(samples) - 2
           
            while(True):
                
                cont = randint(begin, end)
                #print(cont)
                if(cont > 0 and cont < len(samples)-1):
                    if(samples[cont].target != 1):
                        new_samples.append(samples[cont])
                        break
    return create_sequential_bases(new_samples,n_bases)

def channel_selection2(bases_of_samples,selected_channels,fo,alg,result1=None):
    begin = 0
    channels1 = copy.copy(selected_channels)
    if(result1 == None):
        if(sum(selected_channels) == 0):
            for i in range(len(selected_channels)):
                if(selected_channels[i] == 0):
                    selected_channels[i] = 1
                    #result1 = to_evaluate(bases_of_samples,selected_channels)
                    result1 = divide_base_and_execute(bases_of_samples,selected_channels,fo,alg)

                    selected_channels[i] = 0
                    begin = i
                    break
        else:
            #result1 = to_evaluate(bases_of_samples,selected_channels)
            result1 = divide_base_and_execute(bases_of_samples,selected_channels,fo,alg)


    for i in range(begin,len(selected_channels)):
        if(selected_channels[i] == 0):
            selected_channels[i] = 1
            #result = to_evaluate(bases_of_samples,selected_channels)
            result = divide_base_and_execute(bases_of_samples,selected_channels,fo,alg)
            if(result > result1):
                result1 = result
                channels1 = copy.copy(selected_channels)
            selected_channels[i] = 0

    if(sum(channels1) >= 32 or sum(channels1) == sum(selected_channels)):
        return channels1,result1
    else:
        return channel_selection2(bases_of_samples,channels1,fo,alg,result1)
        
if __name__ == "__main__":
    
    create_directories()
    
        
    all_signals = []
    samples = []
    
    
    conf = config.Configuration()   
    conf.auto()
    
    file_samples_saved = "samples_saved_a_01_10_tentarmelhorar.txt"
    result_file = "resultados_a_64_01_10_svm_teste.txt"
    
    if not os.path.exists(DIRECTORY_SIGNALS_CHANNELS+"/"+file_samples_saved):
        
        
        print("Carregando, filtrando e normalizando os sinais")
        if(True):
            for subject in [conf.subject]:
                for i in range(64):
                    #print("valor de i:",i)
                    specific_channel = rf.separate_signals([i],160,subject,DIRECTORY_SIGNALS_CHANNELS+"/"," ")
                    filtered = filter_decimation(specific_channel,conf)                
                    normalized = normalize_base(filtered)                
                    all_signals.append(normalized)
                    
                    
    
        stimulus = rf.process_stimulus_type_file(rf.STIMULUS_FILE_A)
        
        print("Separando os sinais")
        for i in range(len(stimulus)):
        #for i in range(10):        
            samples_signals = []
            
            for j in range(len(all_signals)):
            #for j in range(10):
                samples_signals.append(all_signals[j][i])
                
            samples.append(sample.Sample(samples_signals,stimulus[i]))
            samples_signals = []
       
       
       
       #os.makedirs(DIRECTORY_SIGNALS_CHANNELS+"/samples_saved.txt")
        samples_file = open(DIRECTORY_SIGNALS_CHANNELS+file_samples_saved,"w+") 
        print("Salvando sinais filtrados")
        for i in range(len(samples)):
            samples_file.write(str(i)+"\n")
            samples_file.write(str(samples[i].target)+"\n")
            
            for j in range(len(samples[i].signals)):
                for k in range(len(samples[i].signals[j])):
                    samples_file.write(str(samples[i].signals[j][k])+" ")
                samples_file.write("\n")
        samples_file.close()
    else:
        filename = DIRECTORY_SIGNALS_CHANNELS+file_samples_saved
    
        samples = rf.read_samples_saved(filename)
            
        
    
    print("Criando as bases")
    if(True):
        bases = create_bases_2(samples)
    else:
        bases = create_sequential_bases(samples)
    fo = open(result_file,"w+") 
    fo.write(conf.get_info())
    
    selected = [0]*64
    #selected[59] = 1
    
    if(True):
        channels1,result1 = channel_selection2(bases,selected,fo,conf.alg)
    else:
        if(True):
            selected_channels = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
            result1 = divide_base_and_execute(bases,selected_channels,fo,"lda")

        else:        
            selected_channels = [0]*64
            for i in range(len(selected)):
                if(channels_names[i] in ["C3","C6","CP3","CPz","CP2","AF3","T7","P3","Pz","PO7","PO3","POz","PO8","O1","Iz"]):
                    selected_channels[i] = 1
            result1 = divide_base_and_execute(bases,selected_channels,fo,"svm")

    print(channels1,result1)
    #divide_base_and_execute(bases,selected,fo,"svm")
    
    fo.close()
    
    
    