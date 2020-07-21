# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:24:18 2020

@author: Mark
"""
import sympy as sp
import numpy as np

def Grad(vector, variables):
    Grad_Vector =[[None]*len(variables) for i in range(len(vector))]
    for f_index in range(len(vector)):
        f = vector[f_index]
        for var_index in range(len(variables)):
            vari = variables[var_index] 
            q = sp.diff(f, vari)
            Grad_Vector[f_index][var_index] = q
    return Grad_Vector

class GradOfVector():
    def __init__(self, vector, variables):
        '''
        Input
        ----------
        vector: array
        [f1, f2, f3, ...]
        
        variables: array
        Sympy symbols, [x, y, z, ...]        
        '''
        
        self.Grad_Vector = Grad(vector, variables)
        
    def show(self):
        sp.init_printing(use_latex='mathjax')
        show_Matrix = sp.Matrix(self.Grad_Vector)
        return (show_Matrix)
    
    def evaluate(self, my_dict, show_this = False):
        '''
        Input
        --------
        my_dict: dict
        has the substitutions, {x:2, y:1, z:4, ...}
        
        Return
        --------
        The evaluated matrix
        '''
        if show_this is False:
            return np.array(sp.Matrix(self.Grad_Vector).evalf(subs = my_dict), dtype=np.float64)
        if show_this:
            return sp.Matrix(self.Grad_Vector).evalf(subs = my_dict)


    
    