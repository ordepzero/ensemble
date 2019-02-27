# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:38:58 2017

@author: PeDeNRiQue
"""


class Configuration:
    
    
    
    subject      = None
    
    to_filter    = None
    to_decimate  = None
    filter_ordem = None
    downsampling = None
    lowcut       = None
    highcut      = None
    alg          = None
    filter_type  = None #butter or cheby or analog
    aten         = None
    channels     = None
    
    size_part    = None
    is_test      = None
    n_value      = None
    freq         = None
    params       = None
    
    def __init__(self):
        pass
    

    def auto(self):
        self.subject = "B"
    
        self.to_filter = True
        self.to_decimate = True
        self.filter_ordem = 5
        self.downsampling = 12 #fator 12
        self.lowcut = 0.1
        self.highcut = 10.0
        self.alg = "svm"
        self.filter_type = "cheby" #butter or cheby or analog
        self.aten = 0.5      
        self.size_part = 160
        self.channels= range(64)
        self.is_test = True
        self.n_value = 0
        self.freq = 240
        self.params = {'C': 0.5, 'coef0': 1, 'degree': 2, 'gamma': 1}
    
    def show_configuration(self):
        print("Indivíduo",self.subject)    
        print("Filtrado?",self.to_filter)
        print("Decimação?",self.to_decimate)
        print("Ordem do filtro",self.filter_ordem)
        print("Decimação",self.downsampling)
        print("Lowcut",self.lowcut)
        print("Highcut",self.highcut)
        print("Algoritmo",self.alg)
        print("Filtro",self.filter_type)
        print("Aten",self.aten)
        print("Tamanho do sinal",self.size_part)
        print("Canais",self.channels)
        print("Teste?",self.is_test)
        print("N Value",self.n_value)
        print("Frequencia",self.freq)
        print(str(self.params))
        
    def get_info(self):
        
        infos = ("Indivíduo: "+self.subject+"\n"    
        "Filtrado?: "+str(self.to_filter)+"\n"
        "Decimação?: "+str(self.to_decimate)+"\n"
        "Ordem do filtro: "+str(self.filter_ordem)+"\n"
        "Decimação: "+str(self.downsampling)+"\n"
        "Lowcut: "+str(self.lowcut)+"\n"
        "Highcut: "+str(self.highcut)+"\n"
        "Algoritmo: "+self.alg+"\n"
        "Params: "+str(self.params)+"\n"
        "Filtro: "+self.filter_type+"\n"
        "Aten: "+str(self.aten)+"\n"
        "Tamanho do sinal: "+str(self.size_part)+"\n"
        "Canais: "+str(self.channels)+"\n"
        "Teste?: "+str(self.is_test)+"\n"
        "N Value: "+str(self.n_value)+"\n"
        "Frequencia: "+str(self.freq)+"\n")
        
        return infos