# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 03:57:27 2018

@author: PeDeNRiQue
"""

class Channel:
    
    channels_name = ''
    
    ac_train = []
    ac_test = []
    
    ac_train_m = 0
    ac_test_m = 0
    
    mc = []
    
    
    def __init__(self,channels_name,ac_train,ac_test,ac_train_m,ac_test_m,mc=None):
        self.channels_name = channels_name
    
        self.ac_train = ac_train
        self.ac_test = ac_test
        
        self.ac_train_m = ac_train_m
        self.ac_test_m = ac_test_m
        self.mc = mc
        
    def show(self):
        print("Channel:",self.channels_name)
        print("Ac treino:",self.ac_train)
        print("Ac teste:",self.ac_test)
        print("Ac treino m:",self.ac_train_m)
        print("Ac test m:",self.ac_test_m)
        if(self.mc != None):
            print("Matriz:",self.mc)