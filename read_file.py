# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:24:55 2017

@author: PeDeNRiQue
"""
import sys
import numpy as np
import os
import sample


SIGNAL_FILE_A = "../../bases/data_set_2/subject_A/Subject_A_Train_Signal.txt"
SIGNAL_FILE_B = "../../bases/data_set_2/subject_B/Subject_B_Train_Signal.txt"
STIMULUS_FILE_A = "../../bases/data_set_2/subject_A/Subject_A_Train_StimulusType.txt"
STIMULUS_FILE_B = "../../bases/data_set_2/subject_B/Subject_B_Train_StimulusType.txt"
STIMULUS_CODE_A = "../../bases/data_set_2/subject_A/Subject_A_Train_StimulusCode.txt"
STIMULUS_CODE_B = "../../bases/data_set_2/subject_B/Subject_B_Train_StimulusCode.txt"


SIGNAL_TESTFILE_A = "../../bases/data_set_2/subject_A/Subject_A_Test_Signal.txt"
SIGNAL_TESTFILE_B = "../../bases/data_set_2/subject_B/Subject_B_Test_Signal.txt"
STIMULUS_TESTCODE_A = "../../bases/data_set_2/subject_A/Subject_A_Test_StimulusCode.txt"
STIMULUS_TESTCODE_B = "../../bases/data_set_2/subject_B/Subject_B_Test_StimulusCode.txt"


LAST_STIMULUS_LINE = 7560

def process_stimulus_type_file(filename):
    f = zip(*read_numbers_in_file(filename,"\t",0))    
        
   
    begin = 0
    end = 24
    
    array = []
    
    for line in f:
        #values = []
        begin = 0
        end = 24
        cont = 0
        while True:
            
            
            if(cont % 2 == 0):
                #print(line[begin:end])
                if(1.0 in line[begin:end]):
                    #print("SIM")
                    array.append(1)
                else:
                    #array.append(0)
                    array.append(0)
                begin = end
                end += 18
                
            else:          
                begin = end
                end += 24
            
            cont += 1
            if(end >= LAST_STIMULUS_LINE):#-¨0 para remover os 2,5 segundos que não ocorre nada
                break
        #array.append(values)
        
    return np.array(array)

def process_stimulus_file(filename):
    
    f = zip(*read_numbers_in_file(filename,"\t",0))    
        
   
    begin = 0
    end = 24
    
    array = []
    
    for line in f:
        #values = []
        begin = 0
        end = 24
        cont = 0
        while True:
            
            
            if(cont % 2 == 0):
                #print(line[begin:end])
                array.append(line[begin])
                begin = end
                end += 18
                
            else:          
                begin = end
                end += 24
            
            cont += 1
            if(end >= LAST_STIMULUS_LINE):#-¨0 para remover os 2,5 segundos que não ocorre nada
                break
        #array.append(values)
        
    return np.array(array)

def read_specific_channels_and_save(filename, separator,channels,file_to_save="specific_lines.txt"):
    
    total_channels = 64
    total_letters = 85
    
    for i in channels:
        if(i >= total_channels):
            print("Há 1 ou mais canais fora do intervalo [0-64])")
            sys.exit()
    
    # o último canal não é incluso no experimento
    fo = open(file_to_save, "w+")
    
    numbers_lines = []
    # Vetor contendo todas as linhas do arquivo que serão salvas
    for i in range(total_letters):
        begin = (i * total_channels) 
        [numbers_lines.append(begin+x) for x in channels]
        
    with open(filename) as fp:
        for j, line in enumerate(fp):
            if(j in numbers_lines):
                
                for x in line.split(separator):
                    fo.write(x+" ")
    fo.close()

def separate_each_channel(filename):
    directory = filename
    if not os.path.exists(directory):
        os.makedirs(directory)    
    
    print("Aguarde, separando os sinais em cada arquivo...")    
    
    for channel in range(64):
        file_to_save = directory+"/signal_channel_"+str(channel)+".txt"
        read_specific_channels_and_save(SIGNAL_FILE," ",[channel],file_to_save);
    
    print("Finalizado")  
        
def read_numbers_in_file(filename,separator,length_number):
    array = []  
    
    
    with open(filename,"r") as f:
        content = f.readlines()
        
        for line in content: # read rest of lines
            values = []
            if(len(line) > 2):
                for x in line.split(separator):
                    if(len(x) > length_number):
                        values.append(float(x))
                array.append(values)
    return np.array(array)
    
def read_samples_saved(filename):
    array = []
    with open(filename,"r") as f:
        for line in f:
            data = line.split()
            array.append(list(map(float, data)))
    values = np.array(array)
    cont = 0
    temp_target = 0
    cont_channel = 0
    signals_temp = []
    samples = []
    for i in range(len(values)):            
        if(cont == 0):
            cont = cont + 1
        elif(cont == 1):
            temp_target = values[i][0]
            cont = cont + 1
        elif(cont == 2):
            cont_channel = cont_channel + 1
            signals_temp.append(values[i])
            if(cont_channel == 64):
                cont_channel = 0
                cont = 0
                samples.append(sample.Sample(signals_temp,temp_target))
                signals_temp = []
    return samples
    
def separate_signals(channels,size_part,subject,parent_file="",separator=" "):
    s = parent_file+"signals_channels_"+subject+"/signal_channel_"
    
    for c in channels:
        s = s + str(c)+".txt"
        return separate_signals_parts(s,separator,size_part)
        
    
def separate_signals_parts(filename,separator,size_part):
    length_number = 2
    array = read_numbers_in_file(filename, separator,length_number)
    
    base = []
    for line in range(0,len(array)):
        #print(line)
        begin = 0
        end = 24
        cont = 0
        while True:
            #print(begin,end)
            if(cont % 2 == 0):
                #print(array[line][begin:size_part])
                base.append(array[line][begin:begin+size_part])
                begin = end
                end += 18               
            else:          
                begin = end
                end += 24
            
            cont += 1
            
            if(end >= LAST_STIMULUS_LINE):#-¨0 para remover os 2,5 segundos que não ocorre nada
                break
            
    return base

def separate_signals_parts2(filename,n_char,n_channels,separator,size_part):
    length_number = 2
    array = read_numbers_in_file(filename, separator,length_number)
    
    begin_index_char = n_char * n_channels
    end_index_char = (n_char+1) * n_channels
    array = array[begin_index_char:end_index_char]
    
    base = []
    for line in range(0,len(array)):
        #print(line)
        begin = 0
        end = 24
        cont = 0
        while True:
            #print(begin,end)
            if(cont % 2 == 0):
                #print(array[line][begin:size_part])
                base.append(array[line][begin:begin+size_part])
                begin = end
                end += 18               
            else:          
                begin = end
                end += 24
            
            cont += 1
            
            if(end >= LAST_STIMULUS_LINE):#-¨0 para remover os 2,5 segundos que não ocorre nada
                break
            
    return base
    
def get_letter_signals(signals,letter_index,segment_lenght):
    
    n_channels = 64
    step = 42
    cont = 0
    
    begin = letter_index * n_channels   
    base = []
    
    while(True):
        if(cont >= LAST_STIMULUS_LINE):
            break
            
        base.append(signals[begin:(begin+n_channels),cont:cont+segment_lenght].ravel())
        cont = cont + step
            
    return base

def create_base(signals,subject):
    
    if(subject == "A"):
        stimulus_type_file = STIMULUS_FILE_A
    else:
        stimulus_type_file = STIMULUS_FILE_B
    stimulus_type = process_stimulus_type_file(stimulus_type_file)
    
    base = []
    for i in range(len(signals)):
        base.append(np.append(signals[i],stimulus_type[i]))
    return base
        
    
def givemea_channel_base(channel,size_part,subject):
    
    
    signal_channel = separate_signals([channel],size_part,subject)
    return np.array(create_base(signal_channel,subject))

def teste1():    
#if __name__ == '__main__': 
    
    channels = range(0, 64)
    size_part = 100
    
    SIGNAL_FILE = SIGNAL_FILE_A
    STIMULUS_FILE = STIMULUS_FILE_A
    
    if(True):
        if(False):
            read_specific_channels_and_save(SIGNAL_FILE," ",channels);
        else:
            separate_each_channel("signals_channels_A")
    if(False):
        specific_lines = separate_signals_parts("specific_lines.txt"," ",size_part)
        print(specific_lines)
    elif(False):
        channels = [0]
        size_part = 100
        signal_channel = separate_signals(channels,size_part)
        
    if(False):
        stimulus_type = process_stimulus_type_file(STIMULUS_FILE)
    
    
if __name__ == '__main__': 
    sf = process_stimulus_type_file(STIMULUS_FILE_B)
    sc = process_stimulus_file(STIMULUS_CODE_B)  