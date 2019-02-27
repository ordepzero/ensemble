# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:26:29 2018

@author: PeDeNRiQue
"""

import random
import copy



def avaliate(vector):
    total = 0
    for i in range(len(vector)):
        if(vector[i] == 1):
            if(i % 3 == 0):
                total = total + (i*i)
            elif(i % 5 == 0):
                total = total - (2 * (i*i))
            elif(i % 2 == 0):
                total = total + i
    return total

best_vector = None
best_avaliate = None
selected_channels = [0] * 64

while(sum(selected_channels) < 40):
    
    temp_vector = None
    temp_avaliate = None
    
    for k in range(10):
        
        available_channels = []
        temp_selected_channels = []
        
        for i in range(len(selected_channels)):
            if(selected_channels[i] == 0):
                available_channels.append(i)
        
        for i in range(4):
            temp_item_index = random.choice(range(len(available_channels)))
            temp_selected_channels.append(available_channels[temp_item_index])
            
            temp_item_value = available_channels.pop(temp_item_index)
            
        #print(selected_channels)
        
        for i in range(len(temp_selected_channels)):
            selected_channels[temp_selected_channels[i]] = 1
            
        #temp_vector = copy.copy(vector)
        result = avaliate(selected_channels)
        print(result,sum(selected_channels))
        
        if(temp_vector == None):    
            temp_vector = copy.copy(selected_channels)
            temp_avaliate = result
        elif(result > temp_avaliate):
            temp_vector = copy.copy(selected_channels)
            temp_avaliate = result     
        
        
        for i in range(len(temp_selected_channels)):
            selected_channels[temp_selected_channels[i]] = 0
            
    channels_s = ""
    
    for i in range(len(temp_vector)):
        if(temp_vector[i] == 1):
            channels_s = channels_s + str(i)+" "
    print(channels_s)
    
    selected_channels = copy.copy(temp_vector)
        
        
        
        
        
        
        
        
        
        
            