# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:05:43 2018

@author: PeDeNRiQue
"""

import matplotlib.pyplot as plt


def generate_image(path,signals,title,legend):    
    
    plt.title(title)
    plt.legend(legend, loc='upper left')
    plt.grid(True)
    
    for s in signals:        
        plt.plot(s)
    #plt.show()
    plt.savefig(path+"/"+title+".png")
    
def calculate_mean_signal(samples):
    n_values = len(samples[0].signals[0])
    
    positives = [0]*n_values
    negatives = [0]*n_values
    
    cont_p = 0
    cont_n = 0
    
    for sam in range(len(samples)):
        if(samples[sam].target == 1):
            positives = positives+samples[sam].signals
            cont_p = cont_p+1
        else:
            negatives = negatives+samples[sam].signals
            cont_n = cont_n+1
    
    return positives/cont_p,negatives/cont_n


def plot_signals(signals,legend):    
    
    for s,l in zip(signals,legend):
        plt.plot(s)
        plt.yscale(l)
        plt.title(l)
    plt.grid(True)
    plt.show()
    
def main():
    config_e.create_directories()    
    
    conf = config.Configuration()   
    conf.auto()
    
    mean_signals = []
    
    titles = []
    legends = ["5 P","5 N","8 P","8 N"]

    for s in ["A","B"]:
        for i in [10.0,20.0]:
            for j in [5,8]:
                print(s,i,j)
                
                samples = load_samples(conf)                
                
                
                mean_signals.append(calculate_mean_signal(samples))
                
    
    cont = 0       
    path_to_save_image = config_e.DIRECTORY_IMAGES 
    
    for i in range(0,len(mean_signals),2):
        cont = cont + 1

        s5p = mean_signals[i][0]
        s5n = mean_signals[i][1]
        s8p = mean_signals[i+1][0]
        s8n = mean_signals[i+1][1]
        
        if(i < 4):
            title = "A "
        else:
            title = "B "
        
        for k in range(len(s5p)):
            sigs = []
            
            sigs.append(s5p[k])
            sigs.append(s5n[k])
            sigs.append(s8p[k])
            sigs.append(s8n[k])            
            
            generate_image(path_to_save_image,sigs,title+str(k),legends)