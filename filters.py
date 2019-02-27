# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:23:05 2017

@author: PeDeNRiQue
"""


import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

import create_bases

def plot_signal(signal):
    plt.plot(signal,"r")
    plt.show()
    
def decimate(signal_samples,downsampling=12):
    
    decimated_signal = []
    if(len(signal_samples) < 160):
        print("Numero de amostrar do sinal: "+str(len(signal_samples)))
        return signal_samples
    elif(False):
        for i in range(len(signal_samples)):
            if(i == 0 or i == len(signal_samples)-1 or i%downsampling == 0):
                decimated_signal.append(signal_samples[i])
    elif(True):
        #downsampling na função resample indica o número de amostras do sinal resultante e NÃO o fator de amostragem
        decimated_signal = signal.resample(signal_samples,downsampling)
        '''
        temp_size = len(signal_samples)
        
        while True:
            
            if(False):
                temp_size = int(temp_size / 2)
                signal_samples = signal.decimate(signal_samples,temp_size)
            elif(False):
                decimated_signal = signal.decimate(signal_samples,downsampling)
                break
            elif(False):
                decimated_signal = signal.decimate(signal_samples,downsampling,n=0)
                break
            elif(False):
                decimated_signal = signal.resample(signal_samples,downsampling)
                break
            elif(True):
                decimated_signal = signal.resample(signal_samples,downsampling)
                break
        '''
            
                
    #print(len(decimated_signal))
            
    return np.array(decimated_signal)

def filter_signal(xn,filter_ordem=3,downsampling=14):
    
    b, a = signal.butter(filter_ordem, 0.1)
    
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
    filtered_signal = signal.filtfilt(b, a, xn)
    
    if(False):
        filtered_signal = decimate(filtered_signal)
        
    return filtered_signal
    
def analog_filter(xn,freq,_decimate=True):
    b, a = signal.butter(3, freq/(len(xn)), 'low', analog=True)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
    filtered_signal = signal.filtfilt(b, a, xn)
    
    if(_decimate):
        filtered_signal = decimate(filtered_signal)
    
    return filtered_signal
    
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5,_decimate=False):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.lfilter(b, a, data)
    
    if(_decimate):
        filtered_signal = decimate(filtered_signal)
        
    return filtered_signal
    
def chebyshev1_filter(xn,lowcut, highcut, fs, order,aten,_decimate,downsampling):
    #nyq = 0.5 * fs
    #low = lowcut / nyq
    #high = highcut / nyq
    
    low = 2 * lowcut / fs
    high = 2 * highcut /fs   
    
    b, a = signal.cheby1(order, aten,[low, high],'bandpass')
    
    #filtered_signal = signal.filtfilt(b, a, xn)      
    filtered_signal = signal.lfilter(b, a, xn)
    
    #print(">>>",downsampling)
    if(_decimate):
        #filtered_signal = decimate(filtered_signal,int(downsampling))
        #filtered_signal = signal.decimate(filtered_signal,int(fs/highcut)) #Fator
        #filtered_signal = signal.resample(filtered_signal,10) #Numero de amostras
        filtered_signal = signal.decimate(filtered_signal,int(downsampling)) #Fator
    return filtered_signal   
    
    
    
    
    
    
    
    