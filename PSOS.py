# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:32:52 2020

@author: Mark
"""

import numpy as np
from n_Body_Simulation import Simulation
from functools import partial
from n_body_equations import Two_Electron_Non_Singular
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from tqdm import tqdm


def sigmoid(x):
    return (np.e**x)/(np.e**x + 1)

def eZe_E0(y, pR, Z = 2):
    #Cross section of the eZe space at p_R when E = 0
    return np.sin(2*y)*((2*Z/np.cos(y))+(2*Z/np.sin(y)) - 2/(np.cos(y)+np.sin(y)) - pR**2)**0.5

def get_Part(my_Class, Q_bar, P, T):
    Sim = Simulation(Func_Class = my_Class, time_step = 0.005)
    Sim.AddParts([1000], Q_bar, P, Track_Length = [750])
    Sim.CalcPath(T)
    return Sim.Parts[0]

def find_index(element, my_array):
    #finds the index of element in my_array
    new_array = np.abs(np.array(my_array)-element)
    index = np.argmin(new_array)
    return index

def get_a_and_p_a(p_R, Part):
    index = find_index(p_R, Part.x)
    #return p_a, a
    return Part.z[index], Part.y[index]

def get_S_DEP1_2(par_class, T, theta_ratio, delta):
    b = np.array([0.,0.,0.,0.,0.,-0.707,-0.707,0.,0.])
    a = np.array([0,-0.633,0.633,0,0,3.149,-3.149,0,0])
    
    Q_bar = [0., 0.84089642, 0.84089642, 0.] + a[:4]*delta*(1-theta_ratio) + b[:4]*delta*theta_ratio
    P = [0., 3.74165739, 3.74165739, 0., 0] + a[4:]*delta*(1-theta_ratio) + b[4:]*delta*theta_ratio
    S_DEP1 = get_Part(par_class, Q_bar, P, T)
    
    a = -a

    Q_bar = [0., 0.84089642, 0.84089642, 0.] + a[:4]*delta*(1-theta_ratio) + b[:4]*delta*theta_ratio
    P = [0., 3.74165739, 3.74165739, 0., 0] + a[4:]*delta*(1-theta_ratio) + b[4:]*delta*theta_ratio
    S_DEP2 = get_Part(par_class, Q_bar, P, T)
    return S_DEP1, S_DEP2

class QuickPart():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
        
        

def DisplayGraphs(p_R_array, T = -40, show = False,
                  save = True, file_name = "PSOS_", S_DEP1s = [], S_DEP2s = [],
                  read_file = False, delta = 1e-3, dots = True):
    Z = 2
    step_check = 100
    my_axis = [[-3.5,3.5],[-0.2,2]]
    S_DEP1s = S_DEP1s
    S_DEP2s = S_DEP2s
    
    
    if read_file:
        S_DEP1s = []
        S_DEP2s = []
        #Read File and get needed data
        S_DEP1_name = file_name + "S_DEP1.txt"
        S_DEP2_name = file_name + "S_DEP2.txt"
        
        f_DEP1 = open(S_DEP1_name, 'r')
        counter = 0
        while True:
            str_x = f_DEP1.readline()
            str_y = f_DEP1.readline()
            str_z = f_DEP1.readline()
 
            if (str_x == "") or (str_y=="") or (str_z==""):
                break
                       
            
            x = np.array(str_x.split(','), dtype = float)
            y = np.array(str_y.split(','), dtype = float)
            z = np.array(str_z.split(','), dtype = float)
            
            
            S_DEP1s.append(QuickPart(x, y, z))  
            if counter % step_check == 0:
                print(counter)
            counter +=1
            
        f_DEP1.close()
                       
        
        f_DEP2 = open(S_DEP2_name, 'r')
        counter = 0
        while True:
            str_x = f_DEP2.readline()
            str_y = f_DEP2.readline()
            str_z = f_DEP2.readline()
 
            if (str_x == "") or (str_y=="") or (str_z==""):
                break
                       
            
            x = np.array(str_x.split(','), dtype = float)
            y = np.array(str_y.split(','), dtype = float)
            z = np.array(str_z.split(','), dtype = float)
            

            S_DEP2s.append(QuickPart(x, y, z))
            if counter % step_check == 0:
                print(counter)
            counter +=1      
        f_DEP2.close()
        
        
    # S eZe, TCP
    par_Two_Electrons_Near_Non_Singular = partial(Two_Electron_Non_Singular, Z = Z)
    
    a = np.array([0,1,-1,0,0,0,0,0,0])
    Q_bar = [0., 0.84089642, 0.84089642, 0.] + a[:4]*delta
    P = [0., -3.74165739, -3.74165739, 0., 0] + a[4:]*delta 
    S_TCP1 = get_Part(par_Two_Electrons_Near_Non_Singular, Q_bar, P, T)
        
    a = np.array([0,-1,1,0,0,0,0,0,0])
    Q_bar = [0., 0.84089642, 0.84089642, 0.] + a[:4]*delta
    P = [0., -3.74165739, -3.74165739, 0., 0] + a[4:]*delta    
    S_TCP2 = get_Part(par_Two_Electrons_Near_Non_Singular, Q_bar, P, T)
        
    for p_R in p_R_array:
        p_R_cap = ((2)**0.5*(4*Z-1))**0.5
            
        y = np.linspace(0.001, np.pi/2, 1000)
        pos_x = eZe_E0(y, pR = p_R)
        neg_x = -pos_x
        plt.style.use('dark_background')
        plt.figure(figsize=(7,7))
        plt.title("$p_R = " + str(p_R) + "$")
        plt.xlabel(r"$p_{\alpha}$")
        plt.ylabel(r"$\alpha$")
        plt.plot(pos_x, y, color = 'Yellow')
        plt.plot(neg_x, y, color = 'Yellow')
        my_label = 'Stable eZe, TCP'
        if p_R > -p_R_cap:
            my_label = 'Unstable eZe, TCP, WR'
            
        if p_R < p_R_cap:
            alpha = []
            p_alpha = []
            for S_DEP_index in range(len(S_DEP1s)):
                S_DEP1 = S_DEP1s[S_DEP_index]
                S_DEP2 = S_DEP2s[S_DEP_index]
                temp_p_alpha, temp_alpha = get_a_and_p_a(p_R, S_DEP1)
                alpha.append(temp_alpha)
                p_alpha.append(temp_p_alpha)
                temp_p_alpha, temp_alpha = get_a_and_p_a(p_R, S_DEP2)
                alpha.append(temp_alpha)
                p_alpha.append(temp_p_alpha)
                    
                    
            if dots: 
                plt.scatter(p_alpha[0::2], alpha[0::2], s=1, color = 'Red', label = 'Stable eZe, DEP')
                plt.scatter(p_alpha[1::2], alpha[1::2], s=1, color = 'Green', label = 'Stable eZe, DEP')
                    
            if dots is False:
                plt.plot(p_alpha[0::2], alpha[0::2], color = 'Red', label = 'Stable eZe, DEP')
                plt.plot(p_alpha[1::2], alpha[1::2], color = 'Green', label = 'Stable eZe, DEP')
                
                
            plt.scatter(*get_a_and_p_a(p_R, S_TCP1), color = 'Blue', label = my_label)
            plt.scatter(*get_a_and_p_a(p_R, S_TCP2), color = 'Blue')#, label = my_label)
        plt.xlim(my_axis[0])
        plt.ylim(my_axis[1])
        plt.legend(loc = 'upper left')
        if save:
            plt.savefig('PSOSPNG/PSOS_PR_is_'+str(p_R)+'.png', dpi = 270)
        if show:
            plt.show()
        plt.close()
    

def PSOS(p_R_array, T = -30, theta_ratios = np.linspace(0,1,5), dots = True,
         parallel = False, save = False, save_data_to_file = False,
         file_start = "PSOS_", delta = 1e-3):
    E = 0
    Z = 2
    
    
    theta_ratios = sigmoid(np.array(theta_ratios))
    S_DEP1_name = file_start + "S_DEP1.txt"
    S_DEP2_name = file_start + "S_DEP2.txt"
    
    
    
    #S eZe, DEP
    S_DEP1s = []
    S_DEP2s = []
    par_Two_Electrons_Near_Non_Singular = partial(Two_Electron_Non_Singular, Z = Z)
    par_get_S_DEP1_2 = partial(get_S_DEP1_2,par_class = par_Two_Electrons_Near_Non_Singular,
                               T = T, delta = delta)
    
    if parallel is False:
        for theta_ratio in tqdm(theta_ratios):
            
            
            S_DEP1, S_DEP2 = par_get_S_DEP1_2(theta_ratio = theta_ratio)

            S_DEP1s.append(S_DEP1)
            S_DEP2s.append(S_DEP2)
            
    if parallel:
        size = 200
        num_cores = cpu_count()
        
        for pos in tqdm(range(0, len(theta_ratios), size), position = 0):            
            correct_tr = theta_ratios[pos:pos + size]
            results = Parallel(n_jobs = num_cores)(delayed(par_get_S_DEP1_2)(theta_ratio = theta_ratio)
                                               for theta_ratio in tqdm(correct_tr, position = 1))
            
            if save_data_to_file is False:            
                for S_DEP1, S_DEP2 in results:
                    S_DEP1s.append(S_DEP1)
                    S_DEP2s.append(S_DEP2)
                    
            if save_data_to_file:
                f_DEP1 = open(S_DEP1_name, 'a')
                f_DEP2 = open(S_DEP2_name, 'a')
                for S_DEP1, S_DEP2 in results:
                    f_DEP1.write(",".join(map(str,S_DEP1.x)))
                    f_DEP1.write("\n")
                    f_DEP1.write(",".join(map(str,S_DEP1.y)))
                    f_DEP1.write("\n")
                    f_DEP1.write(",".join(map(str,S_DEP1.z)))
                    f_DEP1.write("\n")
                    
                    f_DEP2.write(",".join(map(str,S_DEP2.x)))
                    f_DEP2.write("\n")
                    f_DEP2.write(",".join(map(str,S_DEP2.y)))
                    f_DEP2.write("\n")
                    f_DEP2.write(",".join(map(str,S_DEP2.z)))
                    f_DEP2.write("\n")
                f_DEP1.close()
                f_DEP2.close()
            
    # S eZe, TCP
    if save_data_to_file is False:            
        DisplayGraphs(p_R_array, T = T, show = True, save = save,
                      file_name = "PSOS_", S_DEP1s = S_DEP1s, S_DEP2s = S_DEP2s,
                      read_file = False, delta = delta, dots = dots)            
                    
        
if __name__ == "__main__":
    if False:
        sections = np.linspace(-12,2,14001)
        PSOS([3., 0,-0.5,-1,-1.5,-2, -2.5, -3.146346284, -4],
             theta_ratios = sections,
             parallel = True, dots = True, save_data_to_file = True,
             file_start = "PSOS_")
        
    if False:
        sections = np.linspace(2,16,14001)
        PSOS([3., 0,-0.5,-1,-1.5,-2, -2.5, -3.146346284, -4],
             theta_ratios = sections,
             parallel = True, dots = True, save_data_to_file = True,
             file_start = "PSOS1_")
        
    
    if True:
        P_r = np.linspace(-6,3.14,458)
        P_r = np.append(P_r, -3.146346284)
        P_r = np.sort(P_r)
        DisplayGraphs(P_r, read_file = True, file_name = "COM_PSOS_")
    