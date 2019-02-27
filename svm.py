# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 13:54:22 2017

@author: PeDeNRiQue
"""

from sklearn import svm

class SVM:
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    mc = []
    my_svm = None
    kernel = None
    param = None
    performance = None
    
    def __init__(self, kernel=None,params=None):
        
        if(kernel != None and params != None):
            self.kernel = kernel
            self.params = params
            self.create(kernel,params)
        else:
            self.create(kernel)
            
    def calculate_performance(self,results,targets,calc_type=None):
        #print(results,targets)
        for i in range(len(results)):
            #print(results[i],targets[i],self.tp,self.fp,self.fn)
            if(results[i] == targets[i]):
                
                if(results[i] != 1):
                    self.tn = self.tn + 1
                elif(results[i] == 1):
                    self.tp = self.tp + 1
            elif(results[i] != 1):
                self.fn = self.fn + 1
            elif(results[i] == 1):
                self.fp = self.fp + 1     
    
        self.performance = self.tp / (self.tp+self.fp+self.fn)
        
        return self.performance
    
    def get_infos(self):
        
        return self.tp,self.tn,self.fp,self.fn
    
    
    def calculate_accuracy(self,results,targets):
        
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        cont = 0
        for i in range(len(results)):
            #print(results[i],targets[i])
            if(results[i] == targets[i]):
                cont += 1
                
                if(results[i] != 1):
                    self.tn = self.tn + 1
                elif(results[i] == 1):
                    self.tp = self.tp + 1
            elif(results[i] != 1):
                self.fn = self.fn + 1
            elif(results[i] == 1):
                self.fp = self.fp + 1                
            
        error = cont / len(results)
        return error
        
    
    def execute_svm(self,train,test,kernel_,params=None):
    
        #inputs = base[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
        #targets = base[:,-1] #COPIAR ULTIMA COLUNA
        #fo.write("Algoritmo: SVM\n")
        #fo.write("Kernel: "+kernel_+"\n")
         
        
        inputs_train = train[:,:-1]
        targets_train = train[:,-1]
        
        inputs_test = test[:,:-1]
        targets_test = test[:,-1]
        
        #print("Treinando")
        if(kernel_ == 'poly'):
            #print(kernel_,str(params))
            clf = self.svm_with_params(kernel_,params)
        else:
            clf = svm.SVC(kernel=kernel_)
        
        clf.fit(inputs_train,targets_train)  
        
        result = clf.predict(inputs_train)    
        train_error = self.calculate_accuracy(result,targets_train)
        self.mc.append(self.get_infos())
    
        result = clf.predict(inputs_test)
        test_error = self.calculate_accuracy(result,targets_test)
        self.mc.append(self.get_infos())
        
        return  train_error,test_error
        
        
    def train(self,train,kernel_='linear',params=None):
        
        inputs_train = train[:,:-1]
        targets_train = train[:,-1]
        
        if(params == None):
            self.my_svm = svm.SVC(kernel=kernel_,probability=True)
        else:
            self.my_svm = self.svm_with_params(kernel_,params)
            
        
        self.my_svm.fit(inputs_train,targets_train) 
        
        return self.my_svm
        
    def test(self,test):
        
        inputs_test = test[:,:-1]
        targets_test = test[:,-1]
        
        result = self.my_svm.predict(inputs_test)    
        test_error = self.calculate_accuracy(result,targets_test)
        self.mc.append(self.get_infos())
        
        return test_error,result
    
    def svm_with_params(self,kernel,params):
        c  = params['C']
        co = params['coef0']
        de = params['degree']
        gm = params['gamma']
        
        return svm.SVC(kernel=kernel,probability=True,C=c,coef0=co,degree=de,gamma=gm)
        
     
    def create(self,kernel=None,params=None):
        
        if(params == None):
            self.my_svm =  svm.SVC(kernel=kernel)
        else:
            c  = params['C']
            co = params['coef0']
            de = params['degree']
            gm = params['gamma']
            
            self.my_svm = svm.SVC(kernel="poly",C=c,coef0=co,degree=de,gamma=gm,decision_function_shape='ovo',probability=False)
            
    def execute_training(self,base):
        
        inputs = base[:,:-1]
        targets = base[:,-1]
    
        
        self.my_svm.fit(inputs,targets) 
    
    def execute_test(self,base):
        
        inputs = base[:,:-1]
        targets = base[:,-1]
        
        if(len(base) == 1):
            results = self.my_svm.predict(inputs)
            error = self.calculate_accuracy(results,targets)
            return error
            
        prob_results = self.my_svm.predict_proba(inputs)
        results = self.my_svm.predict(inputs) 
        performance = self.calculate_performance(results,targets)
        error = self.calculate_accuracy(results,targets)
        
        return results,prob_results,error,performance
        
    
    def decision_function(self,instance):
        
        inputs = instance[:-1]
        
        b_value = self.my_svm.intercept_[0]
        weights_values = self.my_svm.coef_[0]        
        predict_label = self.my_svm.predict([inputs])
        
        total = 0
        
        for i in range(len(inputs)):
            total = total + (weights_values[i] * inputs[i])
            
        return total + b_value
        