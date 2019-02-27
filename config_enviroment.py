# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:28:57 2018

@author: PeDeNRiQue
"""
import os
import datetime

import read_file as rf 

DIRECTORY_BASE = "../../bases"
DIRECTORY_SIGNALS_CHANNELS = DIRECTORY_BASE+"/signals_channels"
DIRECTORY_SIGNALS_CHANNELS_A = DIRECTORY_SIGNALS_CHANNELS+"/signals_channels_A"
DIRECTORY_SIGNALS_CHANNELS_B = DIRECTORY_SIGNALS_CHANNELS+"/signals_channels_B"
DIRECTORY_IMAGES = DIRECTORY_BASE+"/images"


def create_sample_file(conf):
    
    samples_saved_directory = DIRECTORY_BASE+"/samples_saved"    
    
    if not os.path.exists(samples_saved_directory):
        os.makedirs(samples_saved_directory)
        

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)
    
    filename = "ss_seg_m1_1_lfil"+f+".txt"
    
    return samples_saved_directory+"/"+filename


def create_sample_mat_file(conf,prefix=""):
    
    samples_saved_directory = DIRECTORY_BASE+"/samples_saved"    
    
    if not os.path.exists(samples_saved_directory):
        os.makedirs(samples_saved_directory)
        

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)
    
    #filename = "ss_"+prefix+f+".mat"
    filename = prefix+f+".mat"
    
    return samples_saved_directory+"/"+filename
   
def create_result_file(conf): 
    
    result_directory = DIRECTORY_BASE+"/results_"+conf.subject    
    
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
        
    t = str(datetime.datetime.now()) 
    t = t.replace("-","_").replace(":","_").replace(" ","_")[:19]

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)
    f = f+"_"+conf.alg+"_"+t+".txt"
    
    return result_directory+"/poly_"+f

def create_result_mat_file(conf,prefix=""): 
    
    result_directory = DIRECTORY_BASE+"/results_"+conf.subject    
    
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
        
    t = str(datetime.datetime.now()) 
    t = t.replace("-","_").replace(":","_").replace(" ","_")[:19]

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)
    f = f+"_"+conf.alg+"_"+t+".mat"
    
    return result_directory+"/"+prefix+f

def create_result_mat_file_subdic(conf,subdirectory,prefix=""): 
    
    result_directory = DIRECTORY_BASE+"/results_"+conf.subject+"/"+subdirectory    
    
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
        
    t = str(datetime.datetime.now()) 
    t = t.replace("-","_").replace(":","_").replace(" ","_")[:19]

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)
    f = f+"_"+conf.alg+"_"+t+".mat"
    
    return result_directory+"/"+prefix+f

def create_result_pkl_file(conf,index): 
    
    result_directory = DIRECTORY_BASE+"/results_"+conf.subject    
    
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
        
    t = str(datetime.datetime.now()) 
    t = t.replace("-","_").replace(":","_").replace(" ","_")[:19]

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)    
    f = f+"_"+str(index)
    f = f+"_"+conf.alg+"_"+t+".pkl"
    
    return result_directory+"/"+f

def create_result_pkl_file_subdic(conf,index,subdirectory): 
    
    result_directory = DIRECTORY_BASE+"/results_"+conf.subject+"/"+subdirectory    
    
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
        
    t = str(datetime.datetime.now()) 
    t = t.replace("-","_").replace(":","_").replace(" ","_")[:19]

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)    
    f = f+"_"+str(index)
    f = f+"_"+conf.alg+"_"+t+".pkl"
    
    return result_directory+"/"+f

def create_ensemble_file(conf,prefix=""):
    
    samples_saved_directory = DIRECTORY_BASE+"/results_"+conf.subject    
    
    if not os.path.exists(samples_saved_directory):
        os.makedirs(samples_saved_directory)
        

    f = conf.subject
    f = f+"_"+str(conf.filter_ordem)
    f = f+"_"+str(conf.downsampling)
    f = f+"_"+str(conf.lowcut).replace(".","")[0:2]
    f = f+"_"+str(conf.highcut).replace(".","")[0:2]
    f = f+"_"+str(conf.size_part)
    
    filename = prefix+"ens_"+f+".mat"
    
    return samples_saved_directory+"/"+filename

def create_directories():
    
    if not os.path.exists(DIRECTORY_BASE):
        os.makedirs(DIRECTORY_BASE)
    
    if not os.path.exists(DIRECTORY_SIGNALS_CHANNELS):
        os.makedirs(DIRECTORY_SIGNALS_CHANNELS)
    
    if not os.path.exists(DIRECTORY_SIGNALS_CHANNELS_A):
        os.makedirs(DIRECTORY_SIGNALS_CHANNELS_A)
    
    if not os.path.exists(DIRECTORY_SIGNALS_CHANNELS_B):
        os.makedirs(DIRECTORY_SIGNALS_CHANNELS_B)
       
    if not os.path.exists(DIRECTORY_IMAGES):
        os.makedirs(DIRECTORY_IMAGES)

    dic_channels = [DIRECTORY_SIGNALS_CHANNELS_A,DIRECTORY_SIGNALS_CHANNELS_B]
    dic_files    = [rf.SIGNAL_FILE_A,rf.SIGNAL_FILE_B]
    
    #print("Verificando os arquivos de cada canal...\t", end="")
    for dic_channel,dic_file in zip(dic_channels,dic_files):
        for channel in range(64):            
            file_to_save = dic_channel+"/signal_channel_"+str(channel)+".txt"
            if not os.path.exists(file_to_save):
                rf.read_specific_channels_and_save(dic_file," ",[channel],file_to_save);
    #print("[ok]")