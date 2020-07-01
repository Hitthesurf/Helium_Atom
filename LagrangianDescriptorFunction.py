# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 18:26:13 2020

@author: Mark
"""
#Stores the function
import numpy as np


lam = 1
def TheFunction(t,x,y):
    'Can Change at run time by changing value of function'
    dot_x = 2*x
    dot_y = -1*y   
    return np.array([dot_x, dot_y])

