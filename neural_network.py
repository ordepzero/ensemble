# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 07:56:10 2017

@author: PeDeNRiQue
"""

from pybrain.datasets  import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

def calculate_rna_accuracy(data,fnn):
    cont = 0
    for t in data:
        r = fnn.activate(t[0])
        if(r > 0.5):
            r = 1
        else:
            r = 0
        
        if(r == t[1][0]):
            cont += 1
    error = cont / len(data)
    return error

def execute_rna(train,test,fo,n_neuron=50,epoch=100):
    fo.write("Algoritmo: RNA\n")
    fo.write("Neuronios: "+str(n_neuron)+"\n")
    fo.write("Epocas: "+str(epoch)+"\n")

    #inputs = data[:,:-1] #COPIAR TODAS AS COLUNAS MENOS A ULTIMA
    #targets = data[:,-1] #COPIAR ULTIMA COLUNA
    
    n_attributes = len(train[0])-1
    
    
    train_data = ClassificationDataSet(n_attributes, 1,nb_classes=2)
    test_data = ClassificationDataSet(n_attributes, 1,nb_classes=2)
    
    for n in range(len(train)):
        #print(targets[n])
        train_data.addSample( train[n,:-1], [train[n,-1]])
        
    for n in range(len(test)):
        #print(targets[n])
        test_data.addSample( test[n,:-1], [test[n,-1]])
    
    #print(str(len(train_data))+" "+str(len(test_data)))
    #return train_data,test_data
    #train_data._convertToOneOfMany( )
    #test_data._convertToOneOfMany( )
    
    fnn = buildNetwork(train_data.indim,n_neuron, n_neuron, train_data.outdim)
    trainer = BackpropTrainer(fnn, train_data,verbose=False)
    
    epochs = 0
    print("Início do treinamento da RNA");
    for i in range(epoch): 
        if(i % 10 == 0):
            print("Ainda treinando")
        epochs += 1
        trainer.train() 
    
        
    #print (trainer.testOnClassData())
    #print (trainer.testOnData()) 
    
    train_error = calculate_rna_accuracy(train_data,fnn)
    test_error = calculate_rna_accuracy(test_data,fnn)
    
    
    #line_result = str(n_neuron)+"\t"+str(error)+"\t"+str(error_train)+"\t"+str(epochs)
    
    #f.write(line_result+"\n")
    #f.flush()
    return  train_error,test_error
    
    
    
    
    
    
    
    
    
    
    
    
    