# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:52:50 2020

@author: Mark
"""
from n_Body_Simulation import combine, zeta, cot
import numpy as np

class Two_Electrons_Near_McGee():
    def __init__(self, Parts, lam = np.pi/2, H = 0):
        # In this example the electrons have mass 1
        # The nucleus has a large mass(inf)
        self.lam = lam
        self.H = self.H
        
    def W(self, s):
        pass
    
    
    def n_body_system(self, t, *qp):
        dim = 2 #Number of parameters each particle has
        n=1 # Number of particles, as they have been put into one        

        q = np.array(qp[:n])[0]
        p = np.array(qp[n:])[0]
        dot_q = np.zeros([n, dim])
        dot_p = np.zeros([n, dim])   
        
        r = q[0]
        s = q[1]
        
        v = p[0]
        w = p[1]
        
class Two_Electron_Non_Singular():
    def __init__(self, Parts, Z = 2):
        # In this example the electrons have mass 1
        # The nucleus has a large mass(inf)
        self.Z = Z

        #self.convert_cart = self.convert_cart_WR

    def n_body_system_old(self, t, *QP):
        dim = 4
        n = 1
        Z = self.Z
        
        Q = np.array(QP[:n])[0]
        P = np.array(QP[n:])[0]
        dot_Q = np.zeros([n, dim])
        dot_P = np.zeros([n, dim])
        
        
        Q1 = Q[0]
        Q2 = Q[1]
        Q3 = Q[2]
        Q4 = Q[3]
        
        P1 = P[0]
        P2 = P[1]
        P3 = P[2]
        P4 = P[3]
        
        E = P[4]
        
        r1 = Q1**2 + Q2**2
        r2 = Q3**2 + Q4**2
        
        r12 = (r1**2 + r2**2 - 2*(Q1*Q3 + Q2*Q4)**2 + 2*(Q1*Q4-Q2*Q3)**2)**0.5
        
        R = (r1**2 + r2**2)**0.5
        
        Q1_bar = Q1/(R**0.5)
        Q2_bar = Q2/(R**0.5)
        Q3_bar = Q3/(R**0.5)
        Q4_bar = Q4/(R**0.5)
        
        r1_bar = Q1_bar**2 + Q2_bar**2
        r2_bar = Q3_bar**2 + Q4_bar**2
        
        pr_bar = 1/2 * (Q1_bar*P1 + Q2_bar*P2 + Q3_bar*P3 + Q4_bar*P4)
        
        temp1 = 1/4 * Q1 * (P3**2 + P4**2) - 2*Z*Q1 + 2*Q1*r2*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r1*Q1 + (Q4**2-Q3**2)*Q1 - 2*Q2*Q3*Q4)
        dot_P[0][0] = -(temp1 - temp2)
        
        temp1 = 1/4 * Q2 * (P3**2 + P4**2) - 2*Z*Q2 + 2*Q2*r2*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r1*Q2 - (Q4**2-Q3**2)*Q2 - 2*Q1*Q3*Q4)        
        dot_P[0][1] = -(temp1 - temp2)
        
        temp1 = 1/4 * Q3*(P1**2 + P2**2) - 2*Z*Q3 + 2*Q3*r1*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r2*Q3 + (Q2**2 - Q1**2)*Q3 - 2*Q1*Q2*Q4)
        dot_P[0][2] = -(temp1 - temp2)    
        
        temp1 = 1/4 * Q4*(P1**2 + P2**2) - 2*Z*Q4 + 2*Q4*r1*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r2*Q4 - (Q2**2 - Q1**2)*Q4 - 2*Q1*Q2*Q3)
        dot_P[0][3] = -(temp1 - temp2)         
        
        
        dot_Q[0][0] = (1/4)*r2*P1
        dot_Q[0][1] = (1/4)*r2*P2
        dot_Q[0][2] = (1/4)*r1*P3
        dot_Q[0][3] = (1/4)*r1*P4

        '''
        dot_Q[0][0] = ((1/4)*r2_bar*P1 - (1/2)*Q1_bar*r1_bar*r2_bar*pr_bar) * R
        dot_Q[0][1] = ((1/4)*r2_bar*P2 - (1/2)*Q2_bar*r1_bar*r2_bar*pr_bar) * R
        dot_Q[0][2] = ((1/4)*r1_bar*P3 - (1/2)*Q3_bar*r1_bar*r2_bar*pr_bar) * R
        dot_Q[0][3] = ((1/4)*r1_bar*P4 - (1/2)*Q4_bar*r1_bar*r2_bar*pr_bar) * R   
        '''
        return np.array([*dot_Q, *dot_P, E])
    
    def n_body_system(self, t, *QP):
        #With McGee
        #Only works when energy is 0
        dim = 4
        n = 1
        Z = self.Z
    
        
        Q = np.array(QP[:dim], dtype=np.complex)
        P = np.array(QP[dim:], dtype=np.complex)
        dot_Q = np.zeros(dim, dtype=np.complex)
        dot_P = np.zeros(dim, dtype=np.complex)
        dot_E = 0
        
        
        Q1 = Q[0]
        Q2 = Q[1]
        Q3 = Q[2]
        Q4 = Q[3]
        
        P1 = P[0]
        P2 = P[1]
        P3 = P[2]
        P4 = P[3]
        E = P[4]
        
        r1 = Q1**2 + Q2**2
        r2 = Q3**2 + Q4**2
        
        r12 = (r1**2 + r2**2 - 2*(Q1*Q3 + Q2*Q4)**2 + 2*(Q1*Q4-Q2*Q3)**2)**0.5
        
        #R = (r1**2 + r2**2)**0.5
        #E = R*E
               
        pr_bar = 1/2 * (Q1*P1 + Q2*P2 + Q3*P3 + Q4*P4)
        
        temp1 = 1/4 * Q1 * (P3**2 + P4**2) - 2*Z*Q1 + 2*Q1*r2*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r1*Q1 + (Q4**2-Q3**2)*Q1 - 2*Q2*Q3*Q4)
        dot_P[0] = -(temp1 - temp2)
        
        temp1 = 1/4 * Q2 * (P3**2 + P4**2) - 2*Z*Q2 + 2*Q2*r2*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r1*Q2 - (Q4**2-Q3**2)*Q2 - 2*Q1*Q3*Q4)        
        dot_P[1] = -(temp1 - temp2)
        
        temp1 = 1/4 * Q3*(P1**2 + P2**2) - 2*Z*Q3 + 2*Q3*r1*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r2*Q3 + (Q2**2 - Q1**2)*Q3 - 2*Q1*Q2*Q4)
        dot_P[2] = -(temp1 - temp2)    
        
        temp1 = 1/4 * Q4*(P1**2 + P2**2) - 2*Z*Q4 + 2*Q4*r1*(-E + 1/r12)
        temp2 = 2*((r1*r2)/(r12**3))*(r2*Q4 - (Q2**2 - Q1**2)*Q4 - 2*Q1*Q2*Q3)
        dot_P[3] = -(temp1 - temp2)         
        

        dot_Q[0] = ((1/4)*r2*P1 - (1/2)*Q1*r1*r2*pr_bar)
        dot_Q[1] = ((1/4)*r2*P2 - (1/2)*Q2*r1*r2*pr_bar)
        dot_Q[2] = ((1/4)*r1*P3 - (1/2)*Q3*r1*r2*pr_bar)
        dot_Q[3] = ((1/4)*r1*P4 - (1/2)*Q4*r1*r2*pr_bar)  
        
        dot_E = r1*r2*pr_bar*E
        
        return np.array([*dot_Q, *dot_P, dot_E])
    
    def cart_to_QP(self, q, p):
        '''
        Parameters
        ----------
        q : Array
            [[x_1,y_1],[x_2,y_2]]
        p : Array
            [[p_x_1, p_y_1], [p_x_2, p_y_2]]

        Returns
        -------
        Q and P and Q_bar as a turple

        '''
        
        x1 = q[0][0]
        x2 = q[1][0]
        
        y1 = q[0][1]
        y2 = q[1][1]
        
        px1 = p[0][0]
        px2 = p[1][0]
        
        py1 = p[0][1]
        py2 = p[1][1]
        
        Q1 = ((x1+(x1**2 + y1**2)**0.5)/(2))**0.5
        Q2 = ((-x1+(x1**2 + y1**2)**0.5)/(2))**0.5
        Q3 = ((x2+(x2**2 + y2**2)**0.5)/(2))**0.5        
        Q4 = ((-x2+(x2**2 + y2**2)**0.5)/(2))**0.5 
        
        c_y1 = 2*Q1*Q2
        c_y2 = 2*Q3*Q4
        
        if y1 < 0 and c_y1 > 0:
            Q1 = -Q1
        if y1 > 0 and c_y1 < 0:
            Q1 = -Q1
        
        if y2 < 0 and c_y2 > 0:
            Q3 = -Q3
        if y2 > 0 and c_y2 < 0:
            Q3 = -Q3
        
        
        r1 = Q1**2 + Q2**2
        r2 = Q3**2 + Q4**2
        
        R = (r1**2 + r2**2)**0.5        
        
        Q1_bar = Q1/(R**0.5)
        Q2_bar = Q2/(R**0.5)
        Q3_bar = Q3/(R**0.5)
        Q4_bar = Q4/(R**0.5)
        
        P1 = 2*(px1*Q1 + py1*Q2)
        P2 = 2*(py1*Q1 - px1*Q2)
        P3 = 2*(px2*Q3 + py2*Q4)
        P4 = 2*(py2*Q3 - px2*Q4)
        
        E = self.cart_to_energy(q,p)
        E_bar = R*E
        
        return np.array([Q1,Q2,Q3,Q4]), np.array([P1,P2,P3,P4,E_bar]), np.array([Q1_bar,Q2_bar,Q3_bar,Q4_bar]), E
    
    def convert_cart_old(self, Parts):
        
        Q = Parts[0].q
        P = Parts[0].p
        
        Q1 = Q[:,0]
        Q2 = Q[:,1]
        Q3 = Q[:,2]
        Q4 = Q[:,3]
        
        P1 = P[:,0]
        P2 = P[:,1]
        P3 = P[:,2]
        P4 = P[:,3]
        
        x1 = Q1**2 - Q2**2
        x2 = Q3**2 - Q4**2
        
        y1 = 2*Q1*Q2
        y2 = 2*Q3*Q4
        
        new_q1 = combine(1, x1, y1)
        new_q2 = combine(1, x2, y2)
        
        return np.array([new_q1, new_q2])

    def convert_cart(self, Parts):
        
        all_data = Parts[0].all_data.T
        
        Q1 = all_data[0]
        Q2 = all_data[1]
        Q3 = all_data[2]
        Q4 = all_data[3]
        
        P1 = all_data[4]
        P2 = all_data[5]
        P3 = all_data[6]
        P4 = all_data[7]
        
        #x1 = Q1**2 - Q2**2
        #x2 = Q3**2 - Q4**2
        
        #y1 = 2*Q1*Q2
        #y2 = 2*Q3*Q4
        
        r1 = Q1**2 + Q2**2
        r2 = Q3**2 + Q4**2
        
        #R = (r1**2 + r2**2)**0.5
        
        #px1 = (Q1*P1-Q2*P2)/(2*r1)
        #px2 = (Q3*P3-Q4*P4)/(2*r2)
        
        #py1 = (Q2*P1+Q1*P2)/(2*r1)
        #py2 = (Q4*P3+Q3*P4)/(2*r2)
        
        
        alpha = np.arctan(r2/r1)
        #p_r = ((r1*r2)/(2*R)) * (Q1*P1 + Q2*P2 + Q3*P3 + Q4*P4)
        p_r = (1/2)*(Q1*P1 + Q2*P2 + Q3*P3 + Q4*P4)
        p_alpha = (1/2)*((r2/r1)*(Q1*P1+Q2*P2)-(r1/r2)*(Q3*P3+Q4*P4)) * np.sin(2*alpha)
        
        #Times by np.sin(2a) section (9) in Lee Tanner
        
        #new_q1 = combine(1, x1, y1)
        #new_q2 = combine(1, x2, y2)
        
        new_q1 = combine(1, p_r, alpha, p_alpha)
        
        return np.real(np.array([new_q1]))
    
    def calc_ham(self, Parts, pos = 0, t = 0):
        all_data = Parts[0].all_data.T
        
        return np.real(all_data[8][pos])
    
    def cart_to_energy(self, q,p):
        '''
        

        Parameters
        ----------
        q : ndarray
            Holds [[x1,y1],[x2,y2]].
        p : ndarray
            Holds [[px1,py1],[px2,py2]].

        Returns
        -------
        Float
        Returns the energy
        

        '''
        Z = self.Z
        
        x1 = q[0][0]
        y1 = q[0][1]
        
        x2 = q[1][0]
        y2 = q[1][1]
        
        px1 = p[0][0]
        py1 = p[0][1]
        
        px2 = p[1][0]
        py2 = p[1][1] 
        
        p1 = np.linalg.norm([px1,py1])
        p2 = np.linalg.norm([px2,py2])
        
        r1 = np.linalg.norm([x1,y1])
        r2 = np.linalg.norm([x2,y2])

        r12 = np.linalg.norm([x1-x2,y1-y2])

        E = p1**2/2 + p2**2/2 - Z/r1 - Z/r2 + 1/r12
        return E

class Two_Electrons_Near_TCP():
    def __init__(self, Parts, Z = 2, is_eze = True):
        # In this example the electrons have mass 1
        # The nucleus has a large mass(inf)
        self.Z = Z
        if is_eze:
            self.convert_cart = self.convert_cart_eze
        if is_eze is False:
            self.convert_cart = self.convert_cart_WR
    
    def V(self, alpha, theta):
        a = alpha
        O = theta
        Z = self.Z
        my_V = -Z/(np.cos(a)) - Z/(np.sin(a)) + 1/((1-np.sin(2*a)*np.cos(O))**0.5)
        return my_V
    
    def par_alpha_V(self, alpha, theta):
        a = alpha
        O = theta
        Z = self.Z
        
        my_par_alpha_V = -Z*np.sin(a)/((np.cos(a))**2)
        my_par_alpha_V += Z*np.cos(a)/(np.sin(a)**2)
        my_par_alpha_V += (np.cos(2*a)*np.cos(O))/((1-np.sin(2*a)*np.cos(O))**1.5)
        
        return my_par_alpha_V
    
    def par_theta_V(self, alpha, theta):
        a = alpha
        O = theta
        Z = self.Z
        
        my_par_theta_V = -(np.sin(O)*np.sin(2*a))/(2*((1-np.sin(2*a)*np.cos(O))**1.5))
        return my_par_theta_V
    
    def n_body_system(self, t, *qp):        
        dim = 3 #Number of parameters each particle has
        n=1 # Number of particles, as they have been put into one
        
        q = np.array(qp[:n])[0]
        p = np.array(qp[n:])[0]
        dot_q = np.zeros([n, dim])
        dot_p = np.zeros([n, dim])
        
        a = q[0]
        O = q[1]
        R = q[2]
        
        pa = p[0]
        pO = p[1]
        pR = p[2]
        
        
        H_bar = self.calc_H_bar(a, O, pa, pO, pR)
        
        # dot alpha
        dot_q[0][0] = pa 

        # dot theta
        dot_q[0][1] = pO/(np.cos(a)**2*np.sin(a)**2)
        
        # dot R or dot bar H
        dot_q[0][2] = pR*H_bar

        # dot p_alpha
        temp = (pO**2)*((np.cos(a)**2 - np.sin(a)**2)/(np.sin(a)**3*np.cos(a)**3))
        dot_p[0][0] = -(1/2)*pR*pa + temp - self.par_alpha_V(a, O)
        
        # dot p_theta
        dot_p[0][1] = -(1/2)*pR*pO - self.par_theta_V(a, O)
        
        # dot p_R o
        dot_p[0][2] = (1/2)*(pa**2+(pO**2)/(np.cos(a)**2*np.sin(a)**2)) + H_bar
        
        
        
        return np.array([*dot_q, *dot_p])    
    
    def calc_H_bar(self, alpha, theta, dot_alpha, dot_theta, dot_R):
        a = alpha
        O = theta
        
        pa = dot_alpha
        pO = dot_theta
        pR = dot_R
        
        H_bar = (1/2)*(pR**2+pa**2+(pO**2)/(np.cos(a)**2*np.sin(a)**2)) + self.V(a,O)
        return H_bar
    
    def convert_cart_eze(self, Parts):
        q = Parts[0].q
        p = Parts[0].p
        

        a = q[:, 0]
        pa = p[:, 0]
        pR = p[:, 2]
        
        new_x = pR
        new_y = a
        new_z = pa
        
        new_q = combine(1, new_x, new_y, new_z)
        
        return np.array([new_q])
    
    def convert_cart_WR(self, Parts):
        q = Parts[0].q
        p = Parts[0].p
        

        O = q[:, 1]
        pO = p[:, 1]
        pR = p[:, 2]
        
        new_x = pR
        new_y = O
        new_z = pO
        
        new_q = combine(1, new_x, new_y, new_z)
        
        return np.array([new_q])

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
        
        
        dot_q[0][0] = (p[0]*np.sin(a)**2)/m     

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
        
        
        H=1/(2*m)*p[0]**2*np.sin(a)**2+1/(2*m*q[0]**2)*p[1]**2 + m*g*q[0]*cot(a)
        
            
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