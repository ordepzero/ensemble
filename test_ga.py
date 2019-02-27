# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:19:29 2018

@author: PeDeNRiQue
"""

import random
from pyeasyga.pyeasyga import GeneticAlgorithm

import configuration as config
import config_enviroment as config_e
import main as core
import bases
import main_ensemble as me

def fitness (individual, data):
    return me.avaliate_selected_channels(data["base"],individual,0,data["conf"])
    
def create_individual(data):
    return [random.randint(0, 1) for _ in range(64)]

N_TOTAL_CHAR = 85
classifier = "svm"
n_partitions = 17
n_signals_cycle = 12
n_cycles_char = 15
n_signals_char = n_signals_cycle * n_cycles_char
n_char_partition = int(N_TOTAL_CHAR/n_partitions)


config_e.create_directories()    

conf = config.Configuration()   
conf.auto()

result_filename = config_e.create_result_file(conf)
            
samples = core.load_samples2(conf)



bases_of_samples = bases.create_sequential_bases(samples,n_partitions)

data = {"base":bases_of_samples,"conf":conf}

ga = GeneticAlgorithm(data)
ga.create_individual = create_individual
ga.fitness_function = fitness
ga.run()

print(ga.best_individual())


'''
data = [('pear', 50), ('apple', 35), ('banana', 40)]
ga = GeneticAlgorithm(data)

def fitness (individual, data):
    
    
    fitness = 0
    if individual.count(1) == 1:
        for (selected, (fruit, profit)) in zip(individual, data):
            if selected:
                fitness += profit
    return fitness

ga.fitness_function = fitness
ga.run()

print(ga.best_individual())
'''