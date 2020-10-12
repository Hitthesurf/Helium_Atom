import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp

def is_in(data):
    #put the intersection you want here
    if data[2] >= 0: #p_1>=0
        if (-0.01<data[0])and(data[0]<0.01):#q_1 = 0
            return True
    return False


def examplefunc1(t, x1, x2):
    f1 = x2
    f2 = np.cos(x1)
    f3 = 0
    return np.array([f1, f2])


def examplefunc1_with_lin(t, x1, x2, M):
    L = np.matrix([[0, 1],
                   [-np.sin(x1), 0]])

    f1 = x2
    f2 = np.cos(x1)

    M_dot = L*M

    return np.array([f1, f2, M_dot])


def examplefunc2(t, x, y, px, py):

    x_dot = 1*px
    y_dot = 5*py
    px_dot = 1*x
    py_dot = -5*y

    return np.array([x_dot, y_dot, px_dot, py_dot])


def examplefunc2_with_lin(t, x, y, px, py, M):

    L = np.matrix([[0, 0, 1, 0],
                   [0, 0, 0, 5],
                   [1, 0, 0, 0],
                   [0, -5, 0, 0]])

    x_dot = 1*px
    y_dot = 5*py
    px_dot = 1*x
    py_dot = -5*y

    M_dot = L*M

    return np.array([x_dot, y_dot, px_dot, py_dot, M_dot])


def SigFig(num, SF):
    """Significant Figures

    This calculates the rounding of numbers using significant figures,
    num can be complex or in standard form. Can also input an array of elements.
    Also works for matrix and multidimensional arrays

    Parameters
    ----------
    num : float
        The number to be rounded by the significant figures method
    SF : int
        The number of significant figures needed.

    Returns
    -------
    float:
        The number reduced to its significant figures
    numpy array of float:
        Only if entered result is a list or numpy array

    Examples
    --------

    SigFig([0.02156,0.005926,5453000 + 0.02156j],3)
    >>> [0.0216, 0.00593, 5450000 + 0.0216j]

    SigFig(54, 4)
    >>> 54

    SigFig(5.332, 2)
    >>> 5.3

    SigFig(159321.2, 3)
    >>> 159000

    SigFig(0.012345, 4)
    >>> 0.01235

    SigFig(32.3+52.01j, 2)
    >>> 32 + 52j

    SigFig(0.02156, 2)
    >>> 0.022

    SigFig(5453000 + 0.02156j, 2)
    >>> 5500000 + 0.022j

    SigFig(1.3452e-5,3)
    >>> 1.35e-5

    SigFig(0.005926,3)
    >>> 0.00593

    SigFig(0,5) 
    >>> 0

    SigFig(0*1j,5)
    >>> 0
    """

    if type(num) == list or type(num) == np.ndarray or type(num) == np.matrix:
        is_matrix = False
        if type(num) == np.matrix:
            is_matrix = True
            num = np.array(num)

        SF_nums = []
        for selected_num in num:
            SF_nums.append(SigFig(selected_num, SF))

        if is_matrix:
            return np.matrix(SF_nums)
        return np.array(SF_nums)

    real_part = str(num.real)
    imag_part = str(num.imag)

    if num.real == 0 and num.imag == 0:
        return 0

    if num.imag == 0:
        # Real number
        exponent = 'e0'

        # Remove and rember exponent
        if 'e' in real_part:
            exp_pos = real_part.find('e')
            exponent = real_part[exp_pos:]
            real_part = real_part[:exp_pos]

        dot_pos = len(real_part)
        if '.' in real_part:
            dot_pos = real_part.find('.')
            # Remove Dot
            real_part = real_part[:dot_pos] + real_part[dot_pos+1:]

        # Add zeros to end to make sure it doesn't go out of range
        real_part = real_part + '0'*(SF+1)

        for index in range(0, len(real_part)):
            if real_part[index] != '0':
                # Found Start Of Num
                Carry_point = real_part[index + SF]
                real_part = real_part[:index + SF] + '0'*len(real_part)
                # Insert Dot back in
                real_part = real_part[:dot_pos] + '.' + real_part[dot_pos:]
                to_add = 0
                if int(Carry_point) >= 5:
                    # Then it goes up
                    to_add = 10**(dot_pos - (index+SF))

                if to_add != 0:
                    # Since computers make mistakes when adding due to working in base 2 need to add them up
                    # as strings by using decimal
                    my_num = Decimal(real_part) + Decimal(str(to_add))
                    return float(str(my_num)+exponent)

                if to_add == 0:
                    return float(real_part+exponent)

    if num.imag != 0:
        # Complex or imag number
        return SigFig(num.real, SF) + SigFig(num.imag, SF)*1j


class ODEAnalysis():
    '''Plot PhasePlane by using PlotPhasePlane()
        Plot 3D PhaseSpace by using PlotPhaseSpace3D()
        Calculate eingenvalues and eigenvectors of monodromy matrix by using CalcEigen
    '''

    def __init__(self, ODE_To_Use=None, ODELin_To_Use=None, StepOfTime=0.01):
        self.dt = StepOfTime
        self.f = ODE_To_Use  # Current function of RungeKutta
        self.ODE = ODE_To_Use  # ODE system with linearisation
        self.ODELin = ODELin_To_Use  # ODE system without linearisation
        
        
    def StreakLine(self, start_t, stop_t, x_0, plot = False):
        '''
        Plot the streakline(shows flow of dye added to liquid)
        start_t the time at which the dye enters the system
        stop_t the time at which the snapshot is taken
        pos the position where the dye is being entered
        
        could make parralle
        '''
        
        this_dt = self.dt*10
        final_x = []
        for time_start in np.arange(start_t,stop_t,this_dt):
            temp_t, temp_x = self.RungeKutta(stop_t,time_start, x_0)
            final_x.append(temp_x[-1])
            
        final_x = np.array(final_x)
        
        if not(plot):
            return final_x
        
        #Plot graph, only works if 2D
        x = final_x[:,0]
        y = final_x[:,1]
        plt.plot(x, y)
        

    def RungeKutta(self, T, t_0, x_0):
        # As Start at zero so need an extra addition of dt
        N = int(abs((T-t_0)//self.dt)) + 1
        if T > t_0:
            self.dt = abs(self.dt)
        
        if T < t_0:
            self.dt = -abs(self.dt)
        
        
        t = np.zeros(N+1)
        num_x = len(x_0)

        if type(x_0[0]) == np.ndarray or type(x_0[0]) == list:
            x = np.zeros([N+1, num_x, len(x_0[0])])
        else:
            x = np.zeros([N+1, num_x], dtype=np.complex)

        # Initial Conditions
        x[0] = x_0
        t[0] = t_0

        for n in range(0, N-1):
            k1 = (self.dt)*self.f(t[n], *x[n])
            k2 = (self.dt)*self.f(t[n]+self.dt/2, *(x[n]+k1/2))
            k3 = (self.dt)*self.f(t[n]+self.dt/2, *(x[n]+k2/2))
            k4 = (self.dt)*self.f(t[n]+self.dt, *(x[n]+k3))

            x[n+1] = x[n] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            t[n+1] = t[n] + self.dt

        # Last step to make sure it stops on correct value

        last_dt = T - t[N-1]
        k1 = (last_dt)*self.f(t[N-1], *x[N-1])
        k2 = (last_dt)*self.f(t[N-1]+last_dt/2, *(x[N-1]+k1/2))
        k3 = (last_dt)*self.f(t[N-1]+last_dt/2, *(x[N-1]+k2/2))
        k4 = (last_dt)*self.f(t[N-1]+last_dt, *(x[N-1]+k3))

        x[N] = x[N-1] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        t[N] = t[N-1] + last_dt

        #print("Time: " + str(t[-5:]))
        
        self.dt = abs(self.dt)

        return t, x

    def CalcEigen(self, T, x_0):
        # Calculate correct size of M(0)
        self.f = self.MonoDromyFunc
        I = np.identity(len(x_0)).reshape(1, -1)[0]

        t, x = self.RungeKutta(T, 0, [*x_0, *I])
        self.f = self.ODE

        return np.linalg.eig(x[-1][len(x_0):].reshape(-1, len(x_0)))

    def getMonoDromy(self, T, x_0, SF=3):
        self.f = self.MonoDromyFunc
        I = np.identity(len(x_0)).reshape(1, -1)[0]

        t, x = self.RungeKutta(T, 0, [*x_0, *I])
        
        self.f = self.ODE
        
        return SigFig(x[-1][len(x_0):].reshape(-1, len(x_0)), SF)

    def ShowEigen(self, T, x_0, IntegrateOnOrbit=False, epsilon=1e-4, SF=3):
        I = np.identity(len(x_0)).reshape(1, -1)[0]
        eigenvals, eigenvectors = self.CalcEigen(T, x_0)
        self.f = self.ODE
        for index in range(0, len(eigenvals)):
            my_string = "Value " + str(index) + ": " + \
                str(SigFig(eigenvals[index], SF)) + "\n"
            my_string += "Vector " + str(index) + ": \n"
            for eigen_vector_component in eigenvectors[:, index]:
                my_string += str(SigFig(eigen_vector_component, SF)) + ", \n"

            if IntegrateOnOrbit:
                # only compare direction that have been changed.
                diff_before = abs(epsilon*eigenvectors[:, index])
                new_x_0 = x_0 + epsilon*eigenvectors[:, index]
                t, x = self.RungeKutta(T, 0, [*new_x_0])
                final_x = x[-1][:len(x_0)]
                diff_after = abs(x_0 - final_x)

                my_string += "Start: " + str(SigFig(new_x_0, SF)) + "\n"
                my_string += "End: " + str(SigFig(final_x, SF)) + "\n"

                AsyStable = True
                for index in range(0, len(x_0)):
                    if diff_before[index] != 0:
                        # To make sure it is definitely not stable
                        if 1.1*diff_after[index] > diff_before[index]:
                            AsyStable = False

                if AsyStable:
                    my_string += "Asymptotically Stable in this eigenvector direction \n"
                if AsyStable is False:
                    my_string += "Not Asymptotically Stable in this eigenvector direction \n"

            print(my_string)

    def MonoDromyFunc(self, t, *R):
        # Get components and turn into matrix
        ele = np.int(np.round((-1+(1+4*len(R))**0.5)/2))
        new_R = np.array(R).reshape(-1, ele)
        qp = new_R[0]
        M = np.matrix(new_R[1:])
        W = self.ODELin(t, *qp, M)
        new_M = np.array(W[-1]).reshape(1, -1)[0]
        new_qp = W[0:ele]

        return np.array([*new_qp, *new_M])

    def PlotTimeSeries(self, T, t_0, x_0, separate=False):
        t, x = self.RungeKutta(T, t_0, x_0)

        # Plot Graphs
        if separate:
            for index in range(0, len(x[0])):
                plt.plot(t, x[:, index])
                plt.xlabel("Time")
                plt.ylabel("x" + str(index + 1))
                plt.show()

        if separate is False:
            plt.figure(figsize=(10, 5))
            for index in range(0, len(x[0])):
                plt.plot(t, x[:, index].real, label="x" + str(index + 1))

            plt.legend(loc='upper right', fontsize=12)
            plt.title("Time Series")
            plt.xlabel("Time")
            plt.ylabel("x")
            plt.show()

    def PlotPhasePlane(self, xrange, yrange, T, xaxis=1, yaxis=2, t_0=0, gap=0.5):
        # Only works for 2D functions need to set restrictions
        # Restrictions should be in equations
        width = np.arange(xrange[0], xrange[1]+0.01, gap)
        height = np.arange(yrange[0], yrange[1]+0.01, gap)

        # Get all possible cords
        cords = np.array(np.meshgrid(width, height)).T.reshape(-1, 2)

        plt.figure(figsize=(13, 8))
        for cord in cords:
            t, x = self.RungeKutta(T, t_0, cord)
            plt.plot(x[:, (xaxis-1)].real, x[:, (yaxis-1)].real, color="red")

            # Add arrow for direction
            direction = self.f(0, *cord)
            norm_dir = direction/np.linalg.norm(cord)
            plt.arrow(*cord, *(norm_dir*0.01), length_includes_head=False,
                      head_width=0.1, head_length=0.1)

        plt.title("Phase Plane x" + str(xaxis) + " against x" + str(yaxis))
        plt.xlim(*xrange)
        plt.ylim(*yrange)
        plt.show()

    def PlotPhaseSpace3D(self, xrange, yrange, zrange, T, xaxis=1, yaxis=2, zaxis=3, t_0=0, gap=0.5):
        width = np.arange(xrange[0], xrange[1]+0.01, gap)
        height = np.arange(yrange[0], yrange[1]+0.01, gap)
        depth = np.arange(zrange[0], zrange[1]+0.01, gap)

        # Get all possible cords
        cords = np.array(np.meshgrid(width, height, depth)).T.reshape(-1, 3)

        fig = plt.figure(figsize=(13, 8))
        ax = fig.gca(projection='3d')
        for cord in cords:
            t, bold_x = self.RungeKutta(T, t_0, cord)

            # Need to trim bold_x so it is in range
            # Work backwards
            for point_index in range(len(bold_x)-1, -1, -1):
                x = bold_x[point_index][xaxis-1]
                y = bold_x[point_index][yaxis-1]
                z = bold_x[point_index][zaxis-1]
                x_out = not(xrange[0] < x < xrange[1])
                y_out = not(yrange[0] < y < yrange[1])
                z_out = not(zrange[0] < z < zrange[1])

                if (x_out or y_out or z_out):
                    # Remove
                    bold_x = np.delete(bold_x, point_index, 0)

            ax.plot(bold_x[:, xaxis-1].real, bold_x[:, yaxis-1].real, bold_x[:, zaxis-1].real)

        plt.show()
    
    def IntegrateTrajectory(self, T, point, epsilon = 0.1):
        eigenvals, eigenvectors = self.CalcEigen(2*np.pi, point)
        
        stable_intersection = []
        unstable_intersection = []
    
        for index in range(0, len(eigenvals)):
            eigenval = eigenvals[index]
            eigenvector = eigenvectors[:, index]


            if abs(eigenval) <= 0.95:
                #Stable
                #print("Stable: " + str(eigenvector))

                #Same as unstable but going back in time
                self.dt = -abs(self.dt)

                #Other direction
                new_x_0 = point + 0.01*eigenvector
                ts1, xs1 = self.RungeKutta(-10*2*np.pi, 0, new_x_0)

                #Other direction
                new_x_0 = point - 0.01*eigenvector
                ts2, xs2 = self.RungeKutta(-10*2*np.pi, 0, new_x_0)

                self.dt = abs(self.dt)

                xs = [*xs1, *xs2]


                for data in xs:
                    if is_in(data):
                        stable_intersection.append(data)




            if abs(eigenval) >= 1.05:
                #Unstable
                #print("Unstable: " + str(eigenvector))

                #One direction
                new_x_0 = point + 0.01*eigenvector
                tu1, xu1 = self.RungeKutta(10*2*np.pi, 0, new_x_0)

                #Other direction
                new_x_0 = point - 0.01*eigenvector
                tu2, xu2 = self.RungeKutta(10*2*np.pi, 0, new_x_0)

                xu = [*xu1, *xu2]

                for data in xu:
                    if is_in(data):
                        unstable_intersection.append(data)

        return stable_intersection, unstable_intersection
                     
    def Intersection(self, T, x_0, points_needed = 10, parallel = False):
        #Intersection with the is_in function
        t, x = self.RungeKutta(T, 0, x_0)
        points_to_analyse = [x[int((i/points_needed)*len(x))] for i in range(points_needed)]

    
        stable = []
        unstable = []
        
        
        if parallel: # Doesn't work in Juypter Notebook
            pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
            IntegrateTrajectory_partial = partial(self.IntegrateTrajectory, T=T)
            result_list = pool.map(IntegrateTrajectory_partial, points_to_analyse)
            print(result_list)
            

        if parallel is False:           
            for index in range(points_needed):
                temp_stable, temp_unstable = self.IntegrateTrajectory(T, points_to_analyse[index])
                stable = [*stable, *temp_stable]
                unstable = [*unstable, *temp_unstable]
        
        
        if len(stable) > 0:            
            plt.style.use('dark_background')
            cm = plt.cm.get_cmap('winter_r')
            x = np.array(stable)[:,1]
            y = np.array(stable)[:,3]
            z = np.array(stable)[:,2]

            sc = plt.scatter(x, y, c=z, cmap=cm)
            plt.colorbar(sc, label='p_1')
            plt.title('Stable Intersection')
            plt.xlabel("$q_2$")
            plt.ylabel("$p_2$")
            plt.show()
            
        if len(unstable) > 0:            
            plt.style.use('dark_background')
            cm = plt.cm.get_cmap('winter_r')
            x = np.array(unstable)[:,1]
            y = np.array(unstable)[:,3]
            z = np.array(unstable)[:,2]

            sc = plt.scatter(x, y, c=z, cmap=cm)
            plt.colorbar(sc, label='p_1')
            plt.title('Unstable Intersection')
            plt.xlabel("$q_2$")
            plt.ylabel("$p_2$")
            plt.show()
            
        plt.style.use('default')
        
        
        if len(unstable)==0:
            print("No Unstable Intersection")
        
        if len(stable)==0:
            print("No Stable Intersection")        
            
 
    
    

    
        