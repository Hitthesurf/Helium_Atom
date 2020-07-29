import numpy as np
import time

from ODEAnalysis import *
# from n_body_equations import *

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation

import ipyvolume as ipv
import ipywidgets as widgets


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


class c_plot():
    def __init__(self, ax, size=2, text="", colour_axis=False, is_point=False, cmap=0, norm=0, max_dots=300):
        self.ax = ax
        self.x = []
        self.y = []
        self.z = []
        self.size = size
        self.colour_axis = colour_axis
        self.text = text
        self.cmap = cmap
        self.norm = norm
        self.max_dots = max_dots

        if self.colour_axis:
            self.me = self.ax.scatter([], [], c=[], s=self.size**2)
            self.init_data = self.init_data_col
            if is_point:
                self.set_data = self.set_data_col_point
            if is_point is False:
                self.set_data = self.set_data_col_track

        if self.colour_axis is False:
            if is_point:
                self.me, = self.ax.plot([], [], 'bo', ms=self.size)
            if is_point is False:
                self.me, = self.ax.plot([], [], lw=self.size)

            self.init_data = self.init_data_no_col
            self.set_data = self.set_data_no_col

    def init_data_col(self):
        self.me.set_offsets([[]])
        self.me.set_color(self.cmap([]))

    def set_data_col_point(self, x, y, z):
        data = combine(1, x, y)
        self.me.set_offsets(data)
        self.me.set_color(self.cmap(self.norm(z)))

    def set_data_col_track(self, x, y, z):

        step = int(np.ceil(len(x)/self.max_dots))

        data = combine(step, x, y)
        self.me.set_offsets(data)
        self.me.set_color(self.cmap(self.norm(combine(step,z).reshape(1,-1)[0])))

    def init_data_no_col(self):
        self.me.set_data([], [])

    def set_data_no_col(self, x, y, z):
        self.me.set_data(x, y)


class Particle():
    def __init__(self, mass, q_initial, p_initial, Track_Length=500, Size_of_Particle=15, charge=0, Color="Blue"):
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

        # Calculate Hamiltonian
        if self.calc_ham:
            self.CalcHamiltonian(show=False)


    def ShowStatic(self, with_color = False, z_axis = [-15,15], save = False):   
        plt.style.use('dark_background')
        plt.figure(figsize=(7,7))
        for Part in self.Parts:
            if with_color:
                plt.scatter(Part.x, Part.y, c = Part.z, cmap = mpl.cm.winter,
                            vmin = z_axis[0], vmax = z_axis[1], s =12)
                plt.colorbar()
                
                if save:
                    plt.savefig("Static_" + self.sim_name + "_with_color.png")
            else:
                plt.plot(Part.x, Part.y)
                if save:
                    plt.savefig("Static_" + self.sim_name + ".png")
                
        plt.show()



    def ShowAnimation(self, size=15, follow_mass=-1, save=False, link_data=[], z_axis=[-15, 15], with_color=False, max_dots=150):
        '''

        follow_mass
        -3 : The camera remains static
        -2 : The camera follows the largest mass
        -1 : The camera follows the center of mass of the system
        0,1,2, n-1 : The camera follows that particle


        link_data
        links particles together with a line.
        0 means origin
        i means particle i

        examples
        [[0,0]] line drawn between origin and orign(thus no line)

        [[0,1],[1,2]]
        a line drawn from origin to particle 1 
        and a line drawn from 1 to 2

        z_axis
        the colour range of the z axis

        with_color
        uses colour as a 3rd axis

        max_dots
        since the track length is made out of lots of dots to get different colours
        The number of dots can't exceed this, to help combat slow animaitons,
        doesn't matter if saving animation.
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

        num_of_frames = (len(self.Parts[0].x)-1)//self.speed
        plt.style.use('dark_background')
        # The limits on the colour bar, Z limits
        my_norm = mpl.colors.Normalize(vmin=z_axis[0], vmax=z_axis[1])

        if with_color:
            fig, [ax, cax] = plt.subplots(
                1, 2, gridspec_kw={"width_ratios": [50, 1]})
        else:
            fig, ax = plt.subplots(1, 1)

        if with_color:
            cmap = mpl.cm.winter  # Colour you want to use as scale

            cb1 = mpl.colorbar.ColorbarBase(
                cax, cmap=cmap, norm=my_norm, orientation='vertical')
            # The actual color bar
        else:
            cmap = 0

        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)

        # Get total track length
        total_track_length = 0
        for Part in self.Parts:
            total_track_length += Part.tl

        for Part in self.Parts:
            # Main Body
            Part.point = c_plot(
                ax=ax, size=Part.size, colour_axis=with_color, is_point=True, cmap=cmap, norm=my_norm)

            # Track
            dots_in_track = int((Part.tl/total_track_length)*max_dots)
            Part.track_line = c_plot(ax=ax, size=Part.size/5, colour_axis=with_color,
                                     is_point=False, cmap=cmap, norm=my_norm, max_dots=dots_in_track)

        my_links = []
        for link_pos in link_data:
            temp_link, = ax.plot([], [], lw=3)
            my_links.append(temp_link)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        ham_text = None
        if self.calc_ham:
            ham_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        def my_init():

            Points = []
            Tracks = []
            time_text.set_text('')
            if self.calc_ham:
                ham_text.set_text('')

            links_lines = []

            for Part in self.Parts:
                # Main Body
                Part.point.init_data()
                Points.append(Part.point.me)

                # Track
                Part.track_line.init_data()
                Tracks.append(Part.track_line.me)

            # Set up links/bonds
            for my_link in my_links:
                my_link.set_data([], [])
                links_lines.append(my_link)

            return Points, Tracks, links_lines

        def my_animate(i):
            # i represents the frame number
            pos = i*self.speed
            time_text.set_text('Time = '+str(self.t[pos]))

            if self.calc_ham:
                ham_text.set_text('Hamiltonian = ' +
                                  str(self.Hamiltonian[pos]))

            Points = []
            Tracks = []

            links_lines = []
            link_line = None

            for Part in self.Parts:
                # Main Body
                Part.point.set_data(
                    Part.x[pos:pos+1], Part.y[pos:pos+1], Part.z[pos:pos+1])
                Points.append(Part.point.me)

                # Track
                start_pos = max(0, pos-Part.tl)
                Part.track_line.set_data(
                    Part.x[start_pos:pos], Part.y[start_pos:pos], Part.z[start_pos:pos])
                Tracks.append(Part.track_line.me)

            # Draw links
            for link_index in range(len(link_data)):
                my_link = my_links[link_index]
                link_pos = link_data[link_index]
                start_pos = None
                end_pos = None
                elements = len(self.Parts[0].q)
                #link_line = None

                if link_pos[0] == 0:
                    # means origin
                    start_pos = np.zeros([elements, 2])

                else:
                    start_part = self.Parts[link_pos[0]-1]
                    start_pos = combine(1, start_part.x, start_part.y)

                if link_pos[1] == 0:
                    # means origin
                    end_pos = np.zeros([elements, 2])

                else:
                    end_part = self.Parts[link_pos[1]-1]
                    end_pos = combine(1, end_part.x, end_part.y)

                my_link.set_data([start_pos[pos][0], end_pos[pos][0]], [
                                 start_pos[pos][1], end_pos[pos][1]])
                links_lines.append(link_line)

            # Set center of camera
            if follow_mass >= 0:
                # Follow specific mass
                x_mass = self.Parts[follow_mass].x[pos]
                y_mass = self.Parts[follow_mass].y[pos]
                ax.set_xlim(x_mass-size, x_mass+size)
                ax.set_ylim(y_mass-size, y_mass+size)

            if follow_mass == -1:
                # Follows centre of mass of system
                total_mass = 0
                total_x_mass = 0
                total_y_mass = 0
                for Part in self.Parts:
                    total_mass += Part.m
                    total_x_mass += Part.m*Part.x[pos]
                    total_y_mass += Part.m*Part.y[pos]
                x_cen = total_x_mass/total_mass
                y_cen = total_y_mass/total_mass
                ax.set_xlim(x_cen-size, x_cen+size)
                ax.set_ylim(y_cen-size, y_cen+size)

            return Points, Tracks, links_lines

        self.anim = animation.FuncAnimation(fig, my_animate, init_func=my_init,
                                            frames=num_of_frames, interval=40, blit=True)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=25, metadata=dict(
                artist='Mark Pearson'), bitrate=1800)
            self.anim.save(self.sim_name + ".mp4", writer=writer, dpi=300)

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