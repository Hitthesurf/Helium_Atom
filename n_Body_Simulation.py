import numpy as np
import time

from ODEAnalysis import *
from n_Body_Simulation import *

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation


class n_body():
    def __init__(self, mass, G=1, dim=2):
        self.mass = mass
        self.G = G
        self.dim = dim

    def grav_n_body(self, t, *qp):
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


class Particle():
    def __init__(self, mass, q_initial, p_initial, Track_Length=500, Size_of_Particle=15, Color="Blue"):
        self.m = mass  # float
        self.q_0 = q_initial  # array
        self.p_0 = p_initial  # array
        self.tl = Track_Length  # float
        self.size = Size_of_Particle  # float
        self.color = Color  # string
        self.q = []
        self.p = []
        self.track_line = None  # to be a line
        self.point = None  # to be a point


class Simulation():
    def __init__(self, Func_Class=n_body, Speed=10, time_step=0.01, Sim_Name="Simulation_Name"):
        self.Class = Func_Class
        self.speed = Speed
        self.time_step = time_step
        self.Parts = []
        self.t = []
        self.sim_name = Sim_Name

    def AddParts(self, mass, q_initial, p_initial, Track_Length=[500], Size_of_Particle=[15], Color="Blue"):
        for i in range(0, len(mass)):
            self.Parts.append(Particle(
                mass[i], q_initial[i], p_initial[i], Track_Length[i], Size_of_Particle[i], Color))

    def CalcPath(self, T):  # Add save data
        ODE = ODEAnalysis(self.Class, self.time_step)

        # Get initial conditions and mass
        q_0 = []
        p_0 = []
        mass = []

        for Part in self.Parts:
            q_0.append(Part.q_0)
            p_0.append(Part.p_0)
            mass.append(Part.m)

        my_class = n_body(mass)
        ODE = ODEAnalysis(my_class.grav_n_body)
        self.t, x = ODE.RungeKutta(T, 0, [*q_0, *p_0])

        # Save info to parts

        for Part_Index in range(len(self.Parts)):
            self.Parts[Part_Index].q = x[:, Part_Index]
            self.Parts[Part_Index].p = x[:, len(self.Parts) + Part_Index]

    # Only when dim is 2D
    def ShowStatic(self):
        plt.figure(figsize=(9, 8))

        for Part in self.Parts:
            plt.plot(Part.q[:, 0], Part.q[:, 1])
        plt.show()

    # Only when dim is 2D

    def my_init(self):
        line.set_data([], [])
        return line,

    def my_animate(self, i):
        # i represents the frame number
        x = np.linspace(0, 2*np.pi, 50)
        y = np.sin(x-i*0.0628)
        line.set_data(x, y)
        return line,

    def ShowAnimation(self, size=15, follow_mass=-1, save=False):
        '''

        follow_mass
        -3 : The camera remains static
        -2 : The camera follows the largest mass
        -1 : The camera follows the center of mass of the system
        0,1,2, n-1 : The camera follows that particle

        '''
        
        if follow_mass == -2:
            # Follow largest mass
            max_val = 0
            max_pos = -1
            for index in range(0, len(self.Parts)):
                current_mass = self.Parts[index].m
                if current_mass > max_val:
                    max_val = current_mass
                    max_pos = index

            follow_mass = max_pos

        num_of_frames = (len(self.Parts[0].q[:, 0])-1)//self.speed
        plt.style.use('dark_background')
        fig = plt.figure()
        ax = plt.axes(xlim=(-size, size), ylim=(-size, size))

        for Part in self.Parts:
            # Main Body
            Part.point, = ax.plot([], [], 'bo', ms=Part.size)

            # Track
            Part.track_line, = ax.plot([], [], lw=Part.size/5)

        #self.Parts[0].point, = ax.plot([], [], 'bo', ms=30)
        #self.Parts[1].point, = ax.plot([], [], 'bo', ms=15)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def my_init():

            Points = []
            Tracks = []
            time_text.set_text('')

            for Part in self.Parts:
                # Main Body
                Part.point.set_data([], [])
                Points.append(Part.point)

                # Track
                Part.track_line.set_data([], [])
                Tracks.append(Part.track_line)

            return Points, Tracks,

        def my_animate(i):
            # i represents the frame number
            pos = i*self.speed
            time_text.set_text('time = '+str(self.t[pos]))

            Points = []
            Tracks = []

            for Part in self.Parts:
                # Main Body
                Part.point.set_data(Part.q[:, 0][pos], Part.q[:, 1][pos])
                Points.append(Part.point)

                # Track
                start_pos = max(0, pos-Part.tl)
                Part.track_line.set_data(
                    Part.q[:, 0][start_pos:pos], Part.q[:, 1][start_pos:pos])
                Tracks.append(Part.track_line)

            # Set center of camera

            if follow_mass >= 0:
                # Follow specific mass
                x_mass = self.Parts[follow_mass].q[:, 0][pos]
                y_mass = self.Parts[follow_mass].q[:, 1][pos]
                ax.set_xlim(x_mass-size, x_mass+size)
                ax.set_ylim(y_mass-size, y_mass+size)
                
            if follow_mass == -1:
                #Follows centre of mass of system
                total_mass = 0
                total_x_mass = 0
                total_y_mass = 0
                for Part in self.Parts:
                    total_mass += Part.m
                    total_x_mass += Part.m*Part.q[:, 0][pos]
                    total_y_mass += Part.m*Part.q[:, 1][pos]
                x_cen = total_x_mass/total_mass
                y_cen = total_y_mass/total_mass
                ax.set_xlim(x_cen-size, x_cen+size)
                ax.set_ylim(y_cen-size, y_cen+size)

            return Points, Tracks,

        self.anim = animation.FuncAnimation(fig, my_animate, init_func=my_init,
                                            frames=num_of_frames, interval=20, blit=True)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=50, metadata=dict(
                artist='Mark Pearson'), bitrate=1800)
            self.anim.save(self.sim_name + ".mp4", writer=writer, dpi=300)