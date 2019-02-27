# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:10:39 2018

@author: PeDeNRiQue
"""

import math 
import random

def dist(x1,x2,y1,y2):
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist
    
def line(x):
    x = x - 1
    return max(0,(int(x/10)))
    
def column(x):
    x = x - 1
    return max(0,(int(x%10)))


def seq_dist(vect1,vect2):
    total = 0
    for i in vect1:
        for j in vect2:
            x1 = line(i)
            y1 = column(i)
            x2 = line(j)
            y2 = column(j)
            
            total = total + dist(x1,x2,y1,y2)   
    return total

def is_next(v1,v2):
    d = seq_dist([v1],[v2])
    
    if(d < 1.5):
        return True
    else:
        return False
        
def are_next(v1,v2):
    
    for i in v1:
        for j in v2:
            if(is_next(i,j)):
                return True
    return False
    
def generate(last):
    n_ran = []
    
    if(False):
        values = [5,9,15,26,27,28,49,45,58,41]
        
        for i in range(2):
            temp = random.choice(values)
            if(not are_next(last,[temp])):
                n_ran.append(temp)
            
    exc = [8,11,27,35,36,51]
    while(True):
        ran = random.randint(1,60)   
        
        if(ran not in exc and ran not in n_ran):
            if(not are_next(n_ran,[ran])):
                n_ran.append(ran)
        if(len(n_ran) == 6):
            n_ran = [int(x) for x in n_ran]
            n_ran.sort()
            return n_ran
            
   
def new_sequence():   

    last = [8,11,27,35,36,51]
    
    while(True):
        new = generate(last)
        d = seq_dist(last,new)
        
        if(d > 143 and d < 161):
            print(new,d)
            break

for i in range(4):
    new_sequence()
'''
vects = [[8,11,27,35,36,51],[8,11,24,33,37,55]]
#vects = [[8,11,27,35,36,51],[2,28,32,35,54,58],[8,10,18,34,39,56],[1,37,44,46,48,50]]

for i in range(len(vects)-1):
    for j in range(i+1,len(vects)):
        r = seq_dist(vects[i],vects[j])
        print(r)
'''