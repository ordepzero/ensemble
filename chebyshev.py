# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 22:34:41 2017

@author: PeDeNRiQue
"""
import matplotlib.pyplot as plt
import read_file as rf
from scipy import signal
import numpy as np


def signals_means(channel):
    channels = [channel]
    size_part = 160
    
    zeros = [0]*161
    ones = [0]*161
    
    signal_channel = rf.separate_signals(channels,size_part)
    base = rf.create_base(signal_channel,rf.STIMULUS_FILE)
    cont1 = 0
    cont2 = 0
    for i in base:
        if(i[-1] == 1):
            ones = ones + analog_filter(i)
            cont1 += 1
        else:
            zeros = zeros + analog_filter(i)
            cont2 += 1
            if(True and cont1 == 2550 and cont2 == 2550):
                break
        
        
    plt.plot(zeros/len(zeros),"r")
    plt.plot(ones/len(ones),"g")
    plt.show()

    plt.plot(signal.decimate(zeros/len(zeros),14,axis=-1),"b")
    plt.plot(signal.decimate(ones/len(ones),14,axis=-1),"g")
    plt.show()

def plot_signal(signal):
    plt.plot(signal,"r")
    plt.show()
    
def decimate(signal):
    
    decimated_signal = []
    
    for i in range(len(signal)):
        if(i == 0 or i == len(signal)-1 or i%10 == 0):
            decimated_signal.append(signal[i])
            

    return np.array(decimated_signal)
    
def analog_filter(xn):
    b, a = signal.butter(3, 30/160, 'low', analog=True)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
    filtered_signal = signal.filtfilt(b, a, xn)
    
    if(False):
        filtered_signal = decimate(filtered_signal)
    
    return filtered_signal

def filter_signal(xn):
    
    b, a = signal.butter(3, 0.9)
    
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
    filtered_signal = signal.filtfilt(b, a, xn)
    
    if(False):
        filtered_signal = decimate(filtered_signal)
        
    return filtered_signal
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y
    
def chebyshev1_filter(xn, lowcut=0.1, highcut=10.0, fs=240, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = signal.cheby1(1, 0.1,[low, high],'bandpass')
#    zi = signal.lfilter_zi(b, a)
#    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
#    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    return signal.filtfilt(b, a, xn)

def chebyshev2_filter(xn, lowcut=0.1, highcut=10.0, fs=240, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    b, a = signal.cheby2(1, 1,[low, high],'bandpass')
#    zi = signal.lfilter_zi(b, a)
#    z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
#    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    return signal.filtfilt(b, a, xn)
    
if __name__ == "__main__":
    
    subject = "B"
    
    if(False):
        signal_channel = rf.separate_signals([0],160,subject)
        base = rf.create_base(signal_channel,subject)
    
        xn = base[10][:-1]
        
        plot_signal(xn)
        
        y = filter_signal(xn)
        plot_signal(y)
    elif(False):
        signal_channel = rf.separate_signals([0],160,subject)
        base = rf.create_base(signal_channel,subject)
    
        xn = base[10][:-1]
        
        plot_signal(xn)
        
        y = analog_filter(xn)
        plot_signal(y)
    elif(False):
        signal_channel = rf.separate_signals([0],160,subject)
        base = rf.create_base(signal_channel,subject)
    
        xn = base[10][:-1]
        
        plot_signal(xn)
        
        y = butter_bandpass_filter(xn,0.05,10.0,240)
        
        plot_signal(y)

    elif(True):
        signal_channel = rf.separate_signals([0],160,subject)
        base = rf.create_base(signal_channel,subject)
    
        xn = base[10][:-1]
        
        plot_signal(xn)
        
        y1 = chebyshev1_filter(xn,0.1,10.0,240)
        #y2 = chebyshev2_filter(xn,0.1,10.0,240)
        
        plot_signal(y1)
        #plot_signal(y2)
        
    if(False):
        for i in range(64):
            signals_means(i)