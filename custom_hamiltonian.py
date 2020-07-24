# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:26:23 2020

@author: Mark
"""

import sympy as sp
import numpy as np
from matplotlib import pyplot as plt
import Differentiation_Operators as DO


def Norm_Squared(F):
    total = 0
    for F_i in F:
        total += F_i**2
    return sp.simplify(total)

def Show_Latex(my_text, size = 20):
    #1 slash if raw string
    #2 slash if standard string
    plt.figure(figsize=(8,1))
    plt.axis('off')
    plt.text(0, 0.5, my_text, fontsize=size)
    plt.show()
    
#PE
    
def linear_graph_PE(r_cords, mass, g, part):
    grav_PE = 0
    for r_index in range(len(r_cords)):
        grav_PE += mass[r_index]*g*r_cords[r_index][part]    
    return grav_PE

def Get_Ham_Equations(r_cords, variables, dot_variables, return_Ham = False,
                      show_working = False, simplify = False, linear_grav = -1, g=9.81):
    display_str = r''
    for i in range(0,len(r_cords)):
        display_str += 'm_' + str(i+1) + ' '
        
    m = 0
    if len(r_cords) == 1:    
        m = [sp.symbols(display_str)]
    else:
        m = [*sp.symbols(display_str)]
    
    #KE
    dot_r = []
    for r_cord in r_cords:
        current_dot_r = [DO.Chain_Rule(r_cord[i], variables, dot_variables) for i in range(len(r_cord))]
        dot_r.append(current_dot_r)
    
    v_squared = [Norm_Squared(dot_r_i) for dot_r_i in dot_r]
    KE = 0
    for r_index in range(len(r_cords)):
        KE += (1/2)*m[r_index]*v_squared[r_index]
        
    
    #PE
    PE = 0
    #g = sp.symbols(r'g')
    #Linear Grav
    if linear_grav != -1:
        PE += linear_graph_PE(r_cords, m, g, linear_grav)
    

    
    #Calc L
    L = sp.simplify(KE - PE)
    
    if show_working:
        print("Lagrangian")
        Show_Latex("$L = "+str(sp.latex(L))+"$")
    
    
    #Genralized Momentum
    display_str = r''
    for vari_index in range(0,len(variables)):
        q_i = variables[vari_index]
        display_str += 'p_{' + sp.latex(q_i) + '} '

    p_vari_exp = np.zeros(len(variables), dtype = object)
    p_vari = 0
    if len(variables) == 1:    
        p_vari = [sp.symbols(display_str)]
    else:
        p_vari = [*sp.symbols(display_str)]
    
    for vari_index in range(len(variables)):
        p_i = p_vari[vari_index]
        dot_i = dot_variables[vari_index]
        p_vari_exp[vari_index] = sp.simplify(sp.Eq(sp.diff(L, dot_i), p_i))
        
    if show_working:
        print("Generalised Momentum")
        for p_i_exp in p_vari_exp:
            Show_Latex("$"+str(sp.latex(p_i_exp))+"$")
    
    # Solve for the dots
    #May need to change if mult dots in one
    dot_vari_sub = np.zeros(len(variables), dtype = object)
    my_dict = sp.solve(p_vari_exp, dot_variables)

    for vari_index in range(len(variables)):
        dot_i = dot_variables[vari_index]
        dot_vari_sub[vari_index] = my_dict[dot_i]

    if show_working:
        for vari_index in range(len(dot_vari_sub)):
            dot_i = dot_variables[vari_index]
            dot_i_sub = dot_vari_sub[vari_index]
            Show_Latex("$"+sp.latex(dot_i)+"="+str(sp.latex(sp.simplify(dot_i_sub)))+"$")

    #Calc Ham
    H = 0
    for vari_index in range(len(variables)):
        p_i = p_vari[vari_index]
        dot_i = dot_variables[vari_index]
        H += p_i*dot_i
    H -= L
    
    if simplify:
        H = sp.simplify(H)
    
    if show_working:
        print("Hamiltonian")
        Show_Latex("$H = "+str(sp.latex(H))+"$")
    
    
    #Sub the dots into H
    sub_array = []
    for vari_index in range(len(variables)):
        dot_i = dot_variables[vari_index]
        dot_i_sub = dot_vari_sub[vari_index]
        sub_array.append((dot_i, dot_i_sub))
    
    H = H.subs(sub_array)

    if show_working:
        Show_Latex("$H = "+str(sp.latex(H))+"$")    

    if return_Ham:
        return sp.simplify(H)
    
    # Find the ODE System
    dot_q = np.zeros(len(variables), dtype = object)
    dot_p = np.zeros(len(variables), dtype = object)
    
    if simplify is False:
        for vari_index in range(len(variables)):
            p_i = p_vari[vari_index]
            q_i = variables[vari_index]
            dot_q[vari_index] = sp.diff(H, p_i)
            dot_p[vari_index] = -sp.diff(H, q_i)
            
    if simplify:
        for vari_index in range(len(variables)):
            p_i = p_vari[vari_index]
            q_i = variables[vari_index]
            dot_q[vari_index] = sp.simplify(sp.diff(H, p_i))
            dot_p[vari_index] = sp.simplify(-sp.diff(H, q_i))        
    
    
    if show_working:
        print("ODE System")
        for vari_index in range(len(variables)): 
            dot_q_i = dot_q[vari_index]
            dot_p_i = dot_p[vari_index]
            p_i = p_vari[vari_index]
            q_i = variables[vari_index]
            Show_Latex("$\\dot{" +sp.latex(q_i) + "} = "+str(sp.latex(dot_q_i))+"$")
            Show_Latex("$\\dot{" +sp.latex(p_i) + "} = "+str(sp.latex(dot_p_i))+"$")
            
    return dot_q, dot_p

class Custom_System():
    def __init__(self, r_cords, variables, dot_variables, mass, linear_grav = -1, Parts = 0):
        self.r_cords = r_cords
        self.variables = variables
        self.dot_variables = dot_variables
        self.m = mass
        
        self.dot_q, self.dot_p = Get_Ham_Equations(r_cords, variables, 
                                                   dot_variables, return_Ham = False,
                                                   show_working = False, linear_grav = linear_grav)
        Sub_array = []
        #Sub mass in
        display_str = r''
        for i in range(0,len(mass)):
            display_str += 'm_' + str(i+1) + ' '

        m_symbol = 0
        if len(mass) == 1:    
            m_symbol = [sp.symbols(display_str)]
        else:
            m_symbol = [*sp.symbols(display_str)]
            
        for i in range(len(mass)):
            Sub_array.append((m_symbol[i],mass[i]))
        self.dot_q = sp.Matrix(self.dot_q).subs(Sub_array)
        self.dot_p = sp.Matrix(self.dot_p).subs(Sub_array)
        
        
        #Retrieve momentum vars
        display_str = r''
        for vari_index in range(0,len(variables)):
            q_i = variables[vari_index]
            display_str += 'p_{' + sp.latex(q_i) + '} '

        p_vari_exp = np.zeros(len(variables), dtype = object)
        p_vari = 0
        if len(variables) == 1:    
            p_vari = [sp.symbols(display_str)]
        else:
            p_vari = [*sp.symbols(display_str)]
            
        self.p_vari = p_vari
        
        #Define functions as quicker than subbing in
        my_vari = [*variables,*p_vari]
        
        self.f_dot_q = sp.lambdify(my_vari, list(self.dot_q))
        self.f_dot_p = sp.lambdify(my_vari, list(self.dot_p))
        
        
    def n_body_system(self, t, *qp):
        #n = len(self.r_cords)
        #dim = len(self.dot_q)
        
        
        #q = np.array(qp[:dim])
        #p = np.array(qp[dim:])
        
        #dot_q = np.zeros(dim)
        #dot_p = np.zeros(dim)
        

        dot_q = self.f_dot_q(*qp)
        dot_p = self.f_dot_p(*qp)
        
        
        return np.array([*dot_q, *dot_p])
    
    def convert_cart(self,Parts):
        new_q = []
        all_data = Parts[0].all_data
        all_data = all_data.T
        for Part_Index in range(len(self.r_cords)):
            my_Func=np.vectorize(sp.lambdify(self.variables, self.r_cords[Part_Index]), otypes = [np.ndarray])          
            xyz = np.array(my_Func(*all_data[:len(self.variables)]))
            true_xyz = []
            for each_xyz in xyz:
                true_xyz.append(list(each_xyz))
            new_q.append(true_xyz)
        
        return np.real(np.array(new_q))