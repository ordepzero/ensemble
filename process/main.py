# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:55:44 2018

@author: PeDeNRiQue
"""

import os
import copy

from os import listdir
from os.path import isfile, join

import experiment
import channel

def load_src(name, fpath):
    import os, imp
    p = fpath if os.path.isabs(fpath) \
        else os.path.join(os.path.dirname(__file__), fpath)
    return imp.load_source(name, p)

load_src("configuration", "../configuration.py")
import configuration as config


def list_files(directory_name):
    onlyfiles = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]
    
    return onlyfiles
    
def read_file_result(filename):
    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
    return content
    

def fill_header(line):
    
    exp = experiment.Experiment()
    
    if("Inicio:" in line):
        exp.begin_time = line.split("Inicio: ")[1]
    elif("Indivíduo:" in line):
        exp.cfg.subject = line.split("Indivíduo: ")[1]
    elif("Filtrado?:" in line):
        exp.cfg.to_filter = line.split("Filtrado?: ")[1]
    elif("Decimação?" in line):
        exp.cfg.to_decimate = line.split("Decimação?: ")[1]
    elif("Filtro:" in line):
        exp.cfg.filter_type = line.split("Filtro: ")[1] 
    elif("Ordem do filtro:" in line):
        exp.cfg.filter_ordem = line.split("Ordem do filtro: ")[1]
    elif("Lowcut:" in line):
        exp.cfg.lowcut = float(line.split("Lowcut: ")[1])
    elif("Highcut:" in line):
        exp.cfg.highcut = float(line.split("Highcut: ")[1])
    elif("Aten:" in line):
        exp.cfg.aten = float(line.split("Aten: ")[1])
    elif("Decimação:" in line):
        exp.cfg.downsampling = int(line.split("Decimação: ")[1])
    elif("Tamanho do sinal:" in line):
        exp.cfg.size_part = int(line.split("Tamanho do sinal: ")[1])
     
    return exp
    
def fill_channels_results(file,initial_line):
    
    channels = []
    

    control = True
    
    for i in range(initial_line,len(file)):
        line = file[i]
        
        if(control):       
            control = False
            ac_test = []
            ac_train = []
        if("Canais" in line):
            channel_name = line.split(" ")[1:]
            #print(line.split(" ")[1:])         
        elif("Folds" in line):
            pass
        elif("AcTreino" in line):
            parts = line.split("\t")
            ac_test.append(float(parts[0].split(":")[1]))
            ac_train.append(float(parts[1].split(":")[1]))
            
            temp_line = file[i+1]
            
            if("AcMedioTreino" not in temp_line and "AcTreino" not in line):
                mc = [float(tl) for tl in temp_line.split(" ")]
            
        elif("AcMedioTreino" in line):
            parts = line.split("\t")
            ac_test_m = float(parts[0].split(":")[1])
            ac_train_m = float(parts[1].split(":")[1])
            channels.append(channel.Channel(channel_name,ac_test,ac_train,ac_test_m,ac_train_m))
            control = True
        

    return channels
    
def fill_experiment(file,filename):
    
    my_exp = experiment.Experiment()
    my_exp.file_name = filename
    
    cfg = config.Configuration()
    
    i = 0
    for i in range(len(file)):
        if("Canais" in file[i]):
            break
        else:
            line = file[i]            
            
            if("Indivíduo:" in line):
                cfg.subject = line.split("Indivíduo: ")[1]
            elif("Filtrado?:" in line):
                cfg.to_filter = line.split("Filtrado?: ")[1]
            elif("Decimação?" in line):
                cfg.to_decimate = line.split("Decimação?: ")[1]
            elif("Filtro:" in line):
                cfg.filter_type = line.split("Filtro: ")[1] 
            elif("Ordem do filtro:" in line):
                cfg.filter_ordem = line.split("Ordem do filtro: ")[1]
            elif("Lowcut:" in line):
                cfg.lowcut = float(line.split("Lowcut: ")[1])
            elif("Highcut:" in line):
                cfg.highcut = float(line.split("Highcut: ")[1])
            elif("Algoritmo:" in line):
                cfg.alg = line.split("Algoritmo: ")[1]
            elif("Aten:" in line):
                cfg.aten = float(line.split("Aten: ")[1])
            elif("Decimação:" in line):
                cfg.downsampling = int(line.split("Decimação: ")[1])
            elif("Tamanho do sinal:" in line):
                cfg.size_part = int(line.split("Tamanho do sinal: ")[1])

    my_exp.cfg = cfg
    my_exp.channels_results = fill_channels_results(file,i)
    
    return my_exp
    
def list_experiments(exp):
    print(exp.file_name)
    for i in range(1,30):
        chan = exp.best_number(i)
        if(chan != None):
            print(chan.channels_name,";",chan.ac_test_m)

if __name__ == "__main__":
    
    file_name = "../../../bases/results_B"
    files = list_files(file_name)
    
    exps = []    
    
    for file in files:
        s = read_file_result(file_name+"/"+file)
        exp = fill_experiment(s,file)
        exps.append(exp)
        #print(exp.show())
    '''    
    for e in exps:
        print(e.cfg.subject,e.file_name)
        
    print(len(exps))
    '''
    for e in exps:
        list_experiments(e)