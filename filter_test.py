# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:42:22 2017

@author: PeDeNRiQue
"""
import scipy.signal as signal
import matplotlib.pyplot as plt
import random
import numpy as np

amplitude = 10
t = 10
random.seed()
tau = random.uniform(3, 4)
x = np.arange(t)

pure = np.linspace(-1, 1, 160)
noise = np.random.normal(0, 1, pure.shape)
sig = pure + noise

plt.plot(sig)
plt.show()

#s = signal.resample(sig, 14)
#plt.plot(sig)
#plt.show()

s = signal.decimate(sig,12)
plt.plot(s)
plt.show()
#b, a = signal.cheby1(order, aten,[low, high],'bandpass')
    
#filtered_signal = signal.filtfilt(b, a, xn) 
 
