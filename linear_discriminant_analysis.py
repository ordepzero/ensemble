# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:23:44 2017

@author: PeDeNRiQue
"""

import numpy as np

from pybrain.datasets  import ClassificationDataSet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



class LDA:
    
    my_lda = None
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    mc = []
    
    def __init__(self):
        self.create()
        
    def get_infos(self):
        
        return self.tp,self.tn,self.fp,self.fn
    
    def calculate_lda_accuracy(self,data,clf):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        cont = 0
        for t in data:
            r = clf.predict([t[0]])
            if(r > 0.5):
                r = 1
            else:
                r = 0
            
            if(r == t[1]):
                cont += 1
                
                if(r != 1):
                    self.tn = self.tn + 1
                elif(r == 1):
                    self.tp = self.tp + 1
            elif(r != 1):
                self.fn = self.fn + 1
            elif(r == 1):
                self.fp = self.fp + 1
                
        error = cont / len(data)
        return error
    
    def execute_lda(self,train,test):
        
        
        n_attributes = len(train[0])-1
        
        train_data = ClassificationDataSet(n_attributes, 1,nb_classes=2)
        test_data = ClassificationDataSet(n_attributes, 1,nb_classes=2)
        
        for n in range(len(train)):
            #print(targets[n])
            train_data.addSample( train[n,:-1], [train[n,-1]])
            
        for n in range(len(test)):
            #print(targets[n])
            test_data.addSample( test[n,:-1], [test[n,-1]])
            
        clf = LinearDiscriminantAnalysis()
        clf.fit(train_data['input'],np.ravel(train_data['target']))
        
        train_error = self.calculate_lda_accuracy(train_data,clf)
        self.mc.append(self.get_infos())
        test_error = self.calculate_lda_accuracy(test_data,clf)
        self.mc.append(self.get_infos())
        
        return train_error,test_error
    
    def train(self,train):
        
        inputs_train = train[:,:-1]
        targets_train = train[:,-1]
        
        self.my_lda = LinearDiscriminantAnalysis()
        
        self.my_lda.fit(inputs_train,targets_train) 
        
        return self.my_lda
        
    def test(self,test):
        
        inputs_test = test[:,:-1]
        targets_test = test[:,-1]
        
        result = self.my_lda.predict(inputs_test)    
        
        
        return result
    
    def test_proba(self,test):
        
        inputs_test = test[:,:-1]
        
        result = self.my_lda.predict_proba(inputs_test)    
        
        
        return result
    
    def create(self):
        
        self.my_lda = LinearDiscriminantAnalysis()
            
    def execute_training(self,base):
        
        inputs = base[:,:-1]
        targets = base[:,-1]
        
        self.my_lda.fit(inputs,targets) 
    
    def execute_test(self,base):
        
        inputs = base[:,:-1]
        targets = base[:,-1]
        
        result = self.my_lda.predict_proba(inputs)         
        
        return result