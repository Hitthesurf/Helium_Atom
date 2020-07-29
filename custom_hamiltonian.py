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

def zeta(i, j):
    if i > j:
        return True
    elif i <= j:
        return False

def Norm(F):
    return sp.powsimp(Norm_Squared(F)**0.5)

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

def elastic_PE(r_cords, elastic_data):
    ela_PE = 0
    for elastic_band in elastic_data:
        spring_const = elastic_band[1][0]
        natural_L = elastic_band[1][1]
        From = elastic_band[0][0]
        To = elastic_band[0][1]
        
        From_pos = From
        To_pos = To
        
        if type(From) == int:
            From_pos = r_cords[From]
        if type(To) == int:
            To_pos = r_cords[To]
            
        From_pos = np.array(From_pos)
        To_pos = np.array(To_pos)
            
        #Calc distance between From and To
        difference = Norm(From_pos-To_pos)
        ela_PE += sp.simplify((1/2)*spring_const*(difference - natural_L)**2)
    
    return ela_PE

def radial_gravity_PE(r_cords, mass, G):
    m = mass
    grav_PE=0
    n = len(r_cords)
    r_cords = np.array(r_cords)
    for i in range(0,n):                
        for j in range(0,n):
            if zeta(i,j):
                grav_PE -= (G*m[i]*m[j])/(Norm(r_cords[i]-r_cords[j]))
            
    return 

def radial_charge_PE(r_cords, charge, K):
    c = charge
    charge_PE=0
    n = len(r_cords)
    r_cords = np.array(r_cords)
    for i in range(0,n):                
        for j in range(0,n):
            if zeta(i,j):
                charge_PE += (K*c[i]*c[j])/(Norm(r_cords[i]-r_cords[j]))
            
    return charge_PE   

def Get_Ham_Equations(r_cords, variables, return_Ham = False,
                      show_working = False, simplify = False, linear_grav = -1,
                      g=9.81, elastic_data = [], return_Ham_as_well = False, 
                      radial_gravity = False, G = 1, K = 1,
                      radial_charge = False):
    
    #Calculate dot_var
    dot_variables = []
    for var in variables:
        display_str = r'\dot{' + sp.latex(var) + '}'
        dot_variables.append(sp.symbols(display_str))
        
        

    
    #Get mass sysmbols
    display_str = r''
    for i in range(0,len(r_cords)):
        display_str += 'm_' + str(i+1) + ' '
        
    
        
    m = 0
    if len(r_cords) == 1:    
        m = [sp.symbols(display_str)]
    else:
        m = [*sp.symbols(display_str)]
    
    #Get charge symbols
    display_str = r''
    for i in range(0,len(r_cords)):
        display_str += 'c_' + str(i+1) + ' '
        
    
        
    c = 0
    if len(r_cords) == 1:    
        c = [sp.symbols(display_str)]
    else:
        c = [*sp.symbols(display_str)]
    
    
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
        
    if elastic_data != []:
        PE += elastic_PE(r_cords, elastic_data)
    
    if radial_gravity:
        PE += radial_gravity_PE(r_cords, m, G)
    
    if radial_charge:
        PE += radial_charge_PE(r_cords, c, K)
    
    
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
    
    
    if return_Ham_as_well:     
        return dot_q, dot_p, H
    else:
        return dot_q, dot_p

class Custom_System():
    def __init__(self, r_cords, variables, mass, charge = [], linear_grav = -1,
                 g = 9.81, elastic_data = [], radial_gravity = False, G = 1, Parts = 0,
                 K = 1, radial_charge = False):
        '''
        

        Parameters
        ----------
        r_cords : Array of cord
            contains the position cords of the particels.
            example
            [r1]
            [r1,r2,r3]
            
        variables : Array of sympy symbols
            contains the variables that change in the r_cords.
            
            
        mass : Array of masses
        
        charge : Array of the charge for each particle
            The default is [0]*len(mass)
            
        linear_grav : int, optional
            -1 means no linear gravity is considered for the PE
            0 means linear gravity taken in x axis
            1 means linear gravity taken in y axis
            2 means linear gravity taken in z axis
            
            Therefore, applies linear gravity on the ith section of each
            r cord
            The default is -1.
            
        g : float, optional
            The value of acceleration due to constant/linear gravity .
            The default is 9.81.
            
        elastic_data : Array, optional
            Store information on calcualting PE of spring.
            [  [[From,To],[k,l]] 
             ....
             ....
             ....
            ]
            k = spring_constant
            l = natural length
            From = number takes that particles location ie,
             From_pos = r_cords[From]
            if take an array uses that as the position
            Same for To
            
            If no elastic PE then set to []            
            The default is [].
            
        radial_gravity : Boolean, optional
            Use radial gravity
            The default is False
            
        radial_charge : Boolean, optional
            Use radial charge
            The default is False
            
        G : float, optional
            The gravitational constant to use, the one in real life is
            6.67408e-11 m^3kg^-1s^-2
            The default is 1
            
        K : float, optional
            Coulomb constant,
            The default is 1
            
        Parts : int, optional
            Here to make compatible with simulation class, so we can get
            nice animations.
            The default is 0.


        '''
        self.r_cords = r_cords
        self.variables = variables
        
        self.m = mass
        self.H = 0
        self.charge = charge
        if charge == []:
            self.charge = [0]*len(self.m)
        
        
        self.dot_q, self.dot_p, self.H = Get_Ham_Equations(r_cords, variables,
                                                           return_Ham = False,
                                                   show_working = False,
                                                   linear_grav = linear_grav,
                                                   g=g, elastic_data=elastic_data,
                                                   return_Ham_as_well=True,
                                                   radial_gravity = radial_gravity,
                                                   G = G, K = K,
                                                   radial_charge = radial_charge)
        Sub_array = []
        #Sub mass and charge in
        display_str = r''
        for i in range(0,len(mass)):
            display_str += 'm_' + str(i+1) + ' '

        m_symbol = 0
        if len(mass) == 1:    
            m_symbol = [sp.symbols(display_str)]
        else:
            m_symbol = [*sp.symbols(display_str)]
            
        display_str = r''
        for i in range(0,len(mass)):
            display_str += 'c_' + str(i+1) + ' '

        c_symbol = 0
        if len(mass) == 1:    
            c_symbol = [sp.symbols(display_str)]
        else:
            c_symbol = [*sp.symbols(display_str)]
            
        for i in range(len(mass)):
            Sub_array.append((m_symbol[i],mass[i]))
            Sub_array.append((c_symbol[i],self.charge[i]))
        self.dot_q = sp.Matrix(self.dot_q).subs(Sub_array)
        self.dot_p = sp.Matrix(self.dot_p).subs(Sub_array)
        self.H = self.H.subs(Sub_array)
        
        
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
        self.f_H = sp.lambdify(my_vari, self.H)
        
        
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
    
    def calc_ham(self, Parts, pos=0, t=0):
        all_data = Parts[0].all_data
        return np.real(self.f_H(*all_data[pos]))
    
    
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