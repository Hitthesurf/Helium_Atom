# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:52:50 2020

@author: Mark
"""
from n_Body_Simulation import combine, zeta, cot
import numpy as np

class Motion_In_Cone():
    def __init__(self, Parts, g=9.81, alpha = np.pi/4):
        self.g = g
        self.alpha = alpha        
        self.mass = []
        
        for Part in Parts:
            self.mass.append(Part.m)

        
    def n_body_system(self, t, *qp):
        m = self.mass[0]
        g = self.g
        a = self.alpha
        
        dim = 2 #Number of parameters each particle has
        n=1
        
        q = np.array(qp[:n])[0]
        p = np.array(qp[n:])[0]
        dot_q = np.zeros([n, dim])
        dot_p = np.zeros([n, dim])
        
        
        dot_q[0][0] = p[0]/m     

        dot_q[0][1] = (p[1])/(m*(q[0])**2) 
        
        

        dot_p[0][0] = (p[1]**2)/(m*(q[0])**3)-m*g*cot(a)  
        dot_p[0][1] = 0
        
        return np.array([*dot_q, *dot_p])
    
    def convert_cart(self, Parts):
        q = Parts[0].q
        a = self.alpha
        r_ext = q[:, 0]
        theta = q[:,1]
        
        new_x = np.cos(theta)*(r_ext)
        new_y = np.sin(theta)*(r_ext)
        new_z = (r_ext)*cot(a)
        
        new_q = combine(1, new_x, new_y, new_z)
        
        return np.array([new_q])

    def calc_ham(self, Parts, pos = 0,t=0):
        n = len(self.mass)  # Number of particles
        m = self.mass[0]
        g = self.g
        a = self.alpha
        


        q = Parts[0].q[pos]
        p = Parts[0].p[pos]
        
        
        H=1/(2*m)*p[0]**2+1/(2*m*q[0]**2)*p[1]**2 + m*g*q[0]*cot(a)
        
            
        return H 
        

class Spring_Pen():
    def __init__(self, Parts, g = 9.81, length = 1, spring_const = 1000):
        self.dim = 2  #Number of parameters each particle has
        self.g = g
        self.l = length
        self.spring_const = spring_const
        
        self.mass = []

        
        for Part in Parts:
            self.mass.append(Part.m)

        
    def n_body_system(self, t, *qp):
        m = self.mass[0]
        g = self.g
        l = self.l
        sc = self.spring_const
        dim = self.dim
        n=1 #Number of particles
        
        
        q = np.array(qp[:n])[0]
        p = np.array(qp[n:])[0]
        dot_q = np.zeros([n, dim])
        dot_p = np.zeros([n, dim])
        
        
        dot_q[0][0] = p[0]/m     #x

        dot_q[0][1] = (p[1])/(m*(l+q[0])**2) #theta
        
        

        dot_p[0][0] = (p[1]**2)/(m*(l+q[0])**3)-sc*q[0]+m*g*np.cos(q[1])  #x dot * m
        dot_p[0][1] = -np.sin(q[1])*(l+q[0])*m*g       #theta dot * m
        
        return np.array([*dot_q, *dot_p])
        
    def convert_cart(self, Parts):
        q = Parts[0].q
        l = self.l
        x_ext = q[:, 0]
        theta = q[:,1]
        
        new_x = np.sin(theta)*(l + x_ext)
        new_y = -np.cos(theta)*(l + x_ext)
        
        new_q = combine(1, new_x, new_y)
        
        return np.array([new_q])
    
    def calc_ham(self, Parts, pos, t=0):
        m = self.mass[0]
        g = self.g
        l = self.l
        sc = self.spring_const
        
        q, p = Parts[0].q[pos], Parts[0].p[pos]
        
        H = p[0]**2/(2*m)+p[1]**2/(2*m*(l+q[0])**2)+1/2*sc*q[0]**2 -m*g*np.cos(q[1])*(l+q[0])
        
        return H
        



    
class Grav_N_Body():
    def __init__(self, Parts, G= 1, dim=2):
        self.dim = dim  #Number of parameters each particle has
        self.G = G
        
        self.mass = []
        
        for Part in Parts:
            self.mass.append(Part.m)
            
    def n_body_system(self, t, *qp):
        n = len(self.mass)  # Number of particles
        dim = self.dim  # The phase space it is in
        m = self.mass
        G = self.G
        q = np.array(qp[:n])
        p = np.array(qp[n:])
        dot_q = np.zeros([n, dim])
        dot_p = np.zeros([n, dim])

        for i in range(0, n):
            dot_q[i] = p[i]/m[i]
            # sum
            for k in range(0, n):
                if i != k:
                    dot_p[i] += -(G*m[i]*m[k])*(q[i]-q[k]) / \
                        (np.linalg.norm(q[i]-q[k])**3)

        return np.array([*dot_q, *dot_p])
    
    def convert_cart(self, Parts):
        #Already in cart so not much needs to be done here
        new_q = []
        for Part_Index in range(len(Parts)):
            new_q.append(Parts[Part_Index].q)
        return np.array(new_q)
            
    
    def calc_ham(self, Parts, pos = 0,t=0):
        n = len(self.mass)  # Number of particles
        dim = self.dim  # The phase space it is in
        m = self.mass
        G = self.G
        
        q = []
        p = []
        for Part in Parts:
            q.append(Part.q[pos])
            p.append(Part.p[pos])
        
        
        H=0
        #Kenetic energy
        for i in range(0,n):
            H += (np.linalg.norm(p[i])**2)/(2*m[i])       
            
            #PE
            for j in range(0,n):
                if zeta(i,j):
                    H -= (G*m[i]*m[j])/(np.linalg.norm(q[i]-q[j]))
            
        return H  

    
class Charge_N_Body():
    def __init__(self, Parts, K= 1, dim=2):
        self.dim = dim  #Number of parameters each particle has
        self.K = K
        
        self.mass = []
        self.charge = []
        
        for Part in Parts:
            self.mass.append(Part.m)
            self.charge.append(Part.charge)


        
    def n_body_system(self, t, *qp):
        n = len(self.mass)  # Number of particles
        dim = self.dim  # The size of phase space it is in
        m = self.mass
        K = self.K
        c = self.charge
        
        q = np.array(qp[:n])
        p = np.array(qp[n:])
        dot_q = np.zeros([n, dim])
        dot_p = np.zeros([n, dim])
        

        for i in range(0, n):
            dot_q[i] = p[i]/m[i]
            # sum
            for k in range(0, n):
                if i != k:
                    dis_ik = np.linalg.norm(q[i]-q[k])
                    
                    dot_p[i] += (K*c[i]*c[k])*(q[i]-q[k]) / \
                        (dis_ik**3)



        return np.array([*dot_q, *dot_p])
    
    def convert_cart(self, Parts):
        #Already in cart so not much needs to be done here
        new_q = []
        for Part_Index in range(len(Parts)):
            new_q.append(Parts[Part_Index].q)
        return np.array(new_q)

    def calc_ham(self, Parts, pos = 0,t=0):
        n = len(self.mass)  # Number of particles
        dim = self.dim  # The size of phase space it is in
        m = self.mass
        K = self.K
        c = self.charge
        
        q = []
        p = []
        for Part in Parts:
            q.append(Part.q[pos])
            p.append(Part.p[pos])
        
        
        H=0
        #Kenetic energy
        for i in range(0,n):
            H += (np.linalg.norm(p[i])**2)/(2*m[i])       
                
            #PE
            for j in range(0,n):
                if zeta(i,j):
                    H += (K*c[i]*c[j])/(np.linalg.norm(q[i]-q[j]))
            
        return H    
    
    
class Double_Pendulum():
    def __init__(self, Parts, g=9.81, rod_length = [1,0.5]):
        #dim is theta1 and theta2 so fixed at 2
        #n is number of particles, fixed at 2
        self.dim = 1  #Number of parameters each particle has
        self.g = g
        self.rod_length = rod_length
        
        self.mass = []
        
        for Part in Parts:
            self.mass.append(Part.m)
        
    def n_body_system(self, t, *qp):
        #We have q is theta
        #We have p is angular momentum of theta
        n = 2 #len(self.mass)  # Number of particles # 2
        dim = 1  # The parameters each cords will hold
        m = self.mass
        l = self.rod_length
        g = self.g
        
        h = [0,0]
        
        q = np.array(qp[:n])
        p = np.array(qp[n:])
        dot_q = np.zeros([n, dim])
        dot_p = np.zeros([n, dim])
        
        h[0] = (p[0]*p[1]*np.sin(q[0]-q[1]))/(l[0]*l[1]*(m[0]+m[1]*np.sin(q[0]-q[1])**2))
        
        Top = m[1]*l[1]**2*p[0]**2 + (m[0]+m[1])*l[0]**2*p[1]**2 - 2*m[1]*l[0]*l[1]*p[0]*p[1]*np.cos(q[0]-q[1])
        
        h[1] = (Top)/(2*l[0]**2*l[1]**2*(m[0]+m[1]*np.sin(q[0]-q[1])**2)**2)
        
        dot_q[0] = (l[1]*p[0]-l[0]*p[1]*np.cos(q[0]-q[1]))/((l[0]**2)*l[1]*(m[0]+m[1]*np.sin(q[0]-q[1])**2))
    
        dot_q[1] = (-m[1]*l[1]*p[0]*np.cos(q[0]-q[1]) +(m[0]+m[1])*l[0]*p[1])/(m[1]*l[0]*l[1]**2*(m[0]+m[1]*np.sin(q[0]-q[1])**2))
        
        dot_p[0] = -(m[0]+m[1])*g*l[0]*np.sin(q[0])-h[0] +h[1]*np.sin(2*(q[0]-q[1]))
        
        dot_p[1] = -m[1]*g*l[1]*np.sin(q[1]) + h[0] - h[1]*np.sin(2*(q[0]- q[1]))
    
        return np.array([*dot_q, *dot_p])
    
    def convert_cart(self, Parts):
        theta_1 = Parts[0].q[:,0]
        theta_2 = Parts[1].q[:,0]
        l = self.rod_length
            
        x_1 = np.sin(theta_1)*l[0]
        y_1 = -np.cos(theta_1)*l[0]
            
        x_2 = np.sin(theta_1)*l[0]+np.sin(theta_2)*l[1]
        y_2 = -np.cos(theta_1)*l[0]-np.cos(theta_2)*l[1]
        
        new_q1 = combine(1,x_1,y_1)
        new_q2 = combine(1,x_2,y_2)
        
        return np.array([new_q1,new_q2])
    
    def calc_ham(self, Parts, pos = 0,t=0):
        n = len(self.mass)  # Number of particles
        m = self.mass
        l = self.rod_length
        g = self.g

        
        q = []
        p = []
        for Part in Parts:
            q.append(Part.q[pos][0])
            p.append(Part.p[pos][0])
        
        
        H=0
        
        Top = m[1]*l[1]**2*p[0]**2 + (m[0]+m[1])*l[0]**2*p[1]**2 - 2*m[1]*l[0]*l[1]*p[0]*p[1]*np.cos(q[0]-q[1])
        Bottom = 2*m[1]*l[0]**2*l[1]**2*(m[0]+m[1]*np.sin(q[0]-q[1])**2)
        H= Top/Bottom
        H+= -(m[0]+m[1])*g*l[0]*np.cos(q[0])- m[1]*g*l[1]*np.cos(q[1])
        
        return H   