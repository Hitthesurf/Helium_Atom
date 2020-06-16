import numpy as np

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def examplefunc(t, x1, x2):
    f1 = x2
    f2 = np.sin(np.cos(x2)*x1)*np.sin(x1)
    f3 = 0
    return np.array([f1, f2])


class ODEPlotter():
    def __init__(self, FuncToUse=examplefunc, StepOfTime=0.01):
        self.f = FuncToUse
        self.dt = StepOfTime

    def RungeKutta(self, T, t_0, x_0):
        N = int(T//self.dt)
        t = np.zeros(N)
        num_x = len(x_0)
        x = np.zeros([N, num_x])

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

        return t, x

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
                plt.plot(t, x[:, index], label="x" + str(index + 1))

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
            plt.plot(x[:, (xaxis-1)], x[:, (yaxis-1)], color="red")

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

            ax.plot(bold_x[:, xaxis-1], bold_x[:, yaxis-1], bold_x[:, zaxis-1])

        plt.show()