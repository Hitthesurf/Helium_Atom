import numpy as np
import time

from ODEAnalysis import *
# from n_body_equations import *

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation

import ipyvolume as ipv
import ipywidgets as widgets
from mpl_trajectory import trajectory



def cot(x):
    return 1/np.tan(x)


def zeta(i, j):
    if i > j:
        return True
    elif i <= j:
        return False


def combine(step, *cords):
    if step == 0:
        step = 1
    n_cords = np.array(cords).T
    new_c = []
    for i in range(0, len(cords[0]), step):
        Temp = n_cords[i]
        new_c.append(Temp)
    return np.array(new_c)

def time_text(frame, speed, particles):
    pos = frame*speed
    current_time = particles[0].time[pos]
    return f"Time: {current_time}"

def ham_text(frame, speed, particles):
    pos = frame*speed
    current_Ham = particles[0].Hamiltonian[pos]
    return f"Hamiltonian {current_Ham}"



class Particle():
    def __init__(self, mass, q_initial, p_initial, Track_Length=500,
                 Size_of_Particle=15, charge=0, Color="Blue"):
        self.m = mass  # float
        self.q_0 = q_initial  # array
        self.p_0 = p_initial  # array
        self.tl = Track_Length  # float
        self.size = Size_of_Particle  # float
        self.color = Color  # string
        self.q = []
        self.p = []

        self.x = []
        self.y = []
        self.z = []

        self.track_line = None  # to be a line
        self.point = None  # to be a point
        self.charge = charge  # Only needed if using charged n_body problem
        
        self.my_class = 0

class Simulation():
    def __init__(self, Func_Class, Speed=10, time_step=0.01, Sim_Name="Simulation_Name", Calc_Ham=False):
        self.Class = Func_Class
        self.speed = Speed
        self.time_step = time_step
        self.Parts = []
        self.t = []
        self.sim_name = Sim_Name
        self.Hamiltonian = []

        self.calc_ham = Calc_Ham
        self.my_class = 0

    def AddParts(self, mass, q_initial, p_initial, Track_Length=[500], Size_of_Particle=[15], charge=[0], Color="Blue"):
        #Clear at start
        self.Parts = []
        if charge == [0]:
            charge = charge*len(mass)

        for i in range(0, len(mass)):
            self.Parts.append(Particle(
                mass[i], q_initial[i], p_initial[i], Track_Length[i], Size_of_Particle[i], charge[i], Color))

        #Put all data in first Part
        self.Parts[0].initial_data = [*q_initial, *p_initial]

    def CalcHamiltonian(self, show=True):
        calc_energy = self.my_class.calc_ham
        elements = len(self.t)
        H = []
        for pos in range(0, elements):
            H.append(calc_energy(self.Parts, pos, self.t[pos]))
        self.Hamiltonian = np.array(H)
        if show:
            return H[0]
        
    def CalcEquations(self):
        #Calcs class, therefore not run in calc Path as for custom Ham can take
        # a long time
        self.my_class = self.Class(Parts=self.Parts)

    def CalcPath(self, T):

        # Get initial conditions and mass
        q_0 = []
        p_0 = []
        mass = []
        charge = []

        for Part in self.Parts:
            q_0.append(Part.q_0)
            p_0.append(Part.p_0)
            mass.append(Part.m)
            charge.append(Part.charge)
        
        if self.my_class == 0:
            self.my_class = self.Class(Parts=self.Parts)

        # Decide which function to use
        ODE = ODEAnalysis(self.my_class.n_body_system, StepOfTime=self.time_step)

        self.t, x = ODE.RungeKutta(T, 0, self.Parts[0].initial_data)

        # Save info to parts
        for Part_Index in range(len(self.Parts)):
            #Would be better to store all the data in first part
            self.Parts[Part_Index].q = x[:, Part_Index]
            self.Parts[Part_Index].p = x[:, len(self.Parts) + Part_Index]
        self.Parts[0].all_data = x

        
        # Convert Back to cart cords
        new_q = self.my_class.convert_cart(self.Parts)
        has_z = False
        has_y = False
        if len(new_q[0][0]) == 2:
            has_y = True
        
        if len(new_q[0][0]) == 3:
            has_z = True
            has_y = True
        Num_Parts_To_Add = len(new_q) - len(self.Parts)
        #To display extra parts
        for i in range(Num_Parts_To_Add):
            self.AddParts(mass=[None], q_initial=[None], p_initial = [None])
        
        for Part_Index in range(len(self.Parts)):
            self.Parts[Part_Index].x = new_q[Part_Index][:, 0]
            
            if has_y:
                self.Parts[Part_Index].y = new_q[Part_Index][:, 1]
            else:
                self.Parts[Part_Index].y = np.array(
                    [0.0]*len(new_q[Part_Index][:, 0]))                
            if has_z:
                self.Parts[Part_Index].z = new_q[Part_Index][:, 2]
            else:
                self.Parts[Part_Index].z = np.array(
                    [0.0]*len(new_q[Part_Index][:, 0]))
                
        self.Traj_Animation = trajectory(name = self.sim_name)
        

        for Part in self.Parts:
            self.Traj_Animation.plot3D(Part.x, Part.y, Part.z, Size = Part.size,
                                       Particle_Color = Part.color,
                                       Track_Length = Part.tl,
                                       Track_Size = Part.size/3,
                                       Mass = Part.m)

        #Add time to Particles
        self.Traj_Animation.Particles[0].time = self.t
        
        # Calculate Hamiltonian
        if self.calc_ham:
            self.CalcHamiltonian(show=False)

        #Add Ham to Particles
        self.Traj_Animation.Particles[0].Hamiltonian = self.Hamiltonian


    def ShowStatic(self, *args, **kwargs):#with_color = False, z_axis = [-15,15], save = False, s = 12):   
        plt.style.use('dark_background')
        plt.figure(figsize=(7,7))
        self.Traj_Animation.ShowStatic(*args, **kwargs)
                
        plt.show()



    def ShowAnimation(self, *args, **kwargs): # size=15, follow_mass=-1, save=False, link_data=[], z_axis=[-15, 15], with_color=False, max_dots=150):
        plt.style.use('dark_background')
        my_text = [time_text]
        if self.calc_ham:
            my_text.append(ham_text)
        if "text" in kwargs:    
            my_text = [*my_text, *kwargs["text"]]
        kwargs["text"] = my_text
        
        if not("speed" in kwargs):
            kwargs["speed"] = self.speed
        
        self.Traj_Animation.ShowAnimation(*args, **kwargs)

        

    def ShowAnimation3D(self, size=15):
        ipv.figure()
        ipv.style.use("dark")

        x_Part = []
        y_Part = []
        z_Part = []

        for Part in self.Parts:
            temp_x = Part.x
            temp_y = Part.y
            temp_z = Part.z

            x_Part.append(temp_x)
            y_Part.append(temp_y)
            z_Part.append(temp_z)

        x = combine(self.speed*5, *x_Part)
        y = combine(self.speed*5, *y_Part)
        z = combine(self.speed*5, *z_Part)

        u = ipv.scatter(x, y, z, marker="sphere", size=10, color="green")

        ipv.animation_control(u, interval=100)

        ipv.xyzlim(-size, size)
        ipv.show()