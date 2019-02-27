# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 03:57:38 2018

@author: PeDeNRiQue
"""

import copy

def load_src(name, fpath):
    import os, imp
    p = fpath if os.path.isabs(fpath) \
        else os.path.join(os.path.dirname(__file__), fpath)
    return imp.load_source(name, p)

load_src("configuration", "../configuration.py")
import configuration as config


class Experiment:
    file_name = ""
    begin_time = ""
    end_time = ""
    channels_results = []
    cfg = None
    
    def best_number(self,number_of_channels):
        index = None
        for i in range(len(self.channels_results)):
            if(len(self.channels_results[i].channels_name) == number_of_channels):
                index = i
                break
        if(index != None):
            for i in range(index,len(self.channels_results)):
                if(len(self.channels_results[i].channels_name) == number_of_channels and self.channels_results[i].ac_test_m > self.channels_results[index].ac_test_m):
                    index = i
                
        if(index != None):
            return self.channels_results[index]
        else:
            return None
    
    def highest_ac_test_m(self,alist):
        
        for index in range(1,len(alist)):
    
            currentvalue = alist[index]
            position = index
            
            while position>0 and alist[position-1].ac_test_m>currentvalue.ac_test_m:
                alist[position]=alist[position-1]
                position = position-1
            
            alist[position]=currentvalue
        return alist
        
    def highest_ac_train_m(self,alist):
        
        for index in range(1,len(alist)):
    
            currentvalue = alist[index]
            position = index
            
            while position>0 and alist[position-1].ac_train_m>currentvalue.ac_train_m:
                alist[position]=alist[position-1]
                position = position-1
            
            alist[position]=currentvalue
        return alist
        
    def highest_ac_test(self,alist):
        for index in range(1,len(alist)):
    
            currentvalue = alist[index]
            position = index
            
            while position>0 and max(alist[position-1].ac_test)>max(currentvalue.ac_test):
                alist[position]=alist[position-1]
                position = position-1
            
            alist[position]=currentvalue
        return alist
        
        
    def highest_ac_train(self,alist):
        for index in range(1,len(alist)):
    
            currentvalue = alist[index]
            position = index
            
            while position>0 and max(alist[position-1].ac_train)>max(currentvalue.ac_train):
                alist[position]=alist[position-1]
                position = position-1
            
            alist[position]=currentvalue
        return alist
        
    def org(self,criterion):
        #print("OLA")
        
        if(criterion == "ac_test_m"):
            result = self.highest_ac_test_m(copy.deepcopy(self.channels_results))
        elif(criterion == "ac_train_m"):
            result = self.highest_ac_train_m(copy.deepcopy(self.channels_results))
        elif(criterion == "ac_test"):    
            result = self.highest_ac_test(copy.deepcopy(self.channels_results))
        elif(criterion == "ac_train"):
            result = self.highest_ac_train(copy.deepcopy(self.channels_results))
        else:
            
            print("Confira o criterio em 'Experiment.org()'.")
            print("Opções:ac_test_m; ac_train_m; ac_test;ac_train\n")
        return result
        
        
    def my_best_channel(self,criterion_channel):
        result = self.org(criterion=criterion_channel)
        
        #print("Tamanho",len(result))
        
        if(len(result) == 0):
            return None
        
        return result[-1];
    
    def show_configuration(self):
        self.cfg.show_configuration()
        
    def show(self):
        print("Inicio", self.begin_time)
        print("Indivíduo:",self.cfg.subject)        
        print ("Filtro:",self.cfg.to_filter)
        print ("Decimacao:",self.cfg.to_decimate)
        print ("Ordem:",self.cfg.filter_ordem)
        print ("Lowcut:",self.cfg.lowcut)
        print ("Highcut:",self.cfg.highcut)
        print ("Algoritmo:",self.cfg.alg)        
        print ("Classificador:",self.cfg.alg)
        print ("Filtro:",self.cfg.filter_type)
        print ("Atenuacao:",self.cfg.aten)
        print ("Tamanho:",self.cfg.size_part)