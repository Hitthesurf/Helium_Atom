import multiprocessing as mp
import ODEAnalysis as OD
import numpy as np
from functools import partial
from LagrangianDescriptorFunction import TheFunction
import matplotlib.pyplot as plt


           
            
            
class LagrangianDesc():
    def __init__(self, Func,t_0, p, Tau):
        self.t_0 = t_0
        self.p = p
        self.Tau = Tau
        self.dt = 0.01 #Change these dor better acurracy
        self.width = 0.01 #But the smaller they are the slower they become
        self.ODESystem = OD.ODEAnalysis(Func, StepOfTime=self.dt)
        
        #Storing info
        self.pos_times = []
        self.pos_x = []
        
        self.neg_times = []
        self.neg_x = []
    
    
    def setdt(self, new_dt):
        self.dt = new_dt
        self.ODESystem.dt = new_dt
    
    def CalcM(self,x_0):
        b = self.t_0+self.Tau
        a = self.t_0-self.Tau
        self.pos_times, self.pos_x = self.ODESystem.RungeKutta(b, self.t_0, x_0)
        self.neg_times, self.neg_x = self.ODESystem.RungeKutta(a, self.t_0, x_0)
        return self.traprule(a,b, x_0)
    
    def inside_integral(self, t,x_0):


        total = 0
        x = []
        #times, x = self.ODESystem.RungeKutta(t, self.t_0, x_0)
        if t >= self.t_0:
            #Positive times
            arg_t = np.abs(self.pos_times - t)
            index = np.where(arg_t == np.min(arg_t))[0][0]
            x = self.pos_x[index]
        
        if t < self.t_0:
            #Negative times
            arg_t = np.abs(self.neg_times - t)
            index = np.where(arg_t == np.min(arg_t))[0][0]
            x = self.neg_x[index]
            
        
        
        x_dot = self.ODESystem.ODE(t, *x)
        for i in range(len(x_0)):
            temp = (abs(x_dot[i]))**self.p
            total += temp

        return total
    
    def traprule(self, a,b, x_0, in_int = None):
        if in_int == None:
            in_int = self.inside_integral
        
        
        step = self.width
        
        N = int((b-a)/step)
        total = 0
        for k in range(-1,N+1):
            total += in_int(a+k*step, x_0)
        total -=(in_int(a,x_0)+in_int(b,x_0))/2
        total = step*total
        return total


def CalcMInd(x_0, t_0, p, Tau, dt, w):
    LD = LagrangianDesc(TheFunction, t_0, p, Tau)
    LD.setdt(dt)
    LD.width = w
    return LD.CalcM(x_0)


class LagrangianDescPlot():
    def __init__(self,t_0, p, Tau, parallel = False, cores = -1, save = True):
        self.t_0 = t_0
        self.p = p
        self.Tau = Tau
        self.dt = 0.01 #Change these for better acurracy
        self.width = 0.1
        self.parallel = parallel
        self.save = save
        if cores == -1:
            self.cores = mp.cpu_count()
        else:
            self.cores = cores
        
        
    
    
    
    def Calc_Results(self, cords):
        
        newCalcMInd = partial(CalcMInd, t_0 = self.t_0, p=self.p, Tau=self.Tau, dt=self.dt, w=self.width)
        results = []
        if self.parallel is False:
            results = np.array(list(map(newCalcMInd,cords)))
                
        if self.parallel:
            p = mp.Pool(processes=self.cores)
            results= np.array(list(p.map(newCalcMInd,cords)))
            
        return results
        
    
    def PlotLD(self, x_axis,y_axis, xres, yres):
        x_min, x_max, y_min, y_max = x_axis[0], x_axis[1], y_axis[0], y_axis[1]
            
        cords = np.array([[[x, y] for x in np.linspace(x_min, x_max, num=xres)]
                              for y in np.linspace(y_min, y_max, num=yres)])
        cords = cords.reshape(-1,2)
        
        #Do map
        results = self.Calc_Results(cords).reshape(yres, xres)
            
        
    
        # Make the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(results, interpolation="bicubic", origin="lower",
                       extent=[x_min, x_max, y_min, y_max])
        plt.colorbar()
        plt.title(r" Lagrangian Descriptor Plot")
        plt.xlabel(r"Description of $x$ coordinate (units)")
        plt.ylabel(r"Description of $y$ coordinate (units)")
        if self.save:
            plt.savefig("My_PNG")
        plt.show()    