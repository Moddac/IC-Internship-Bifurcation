"""
Attempt to solve interacting particle SDE with Euler-Maruyama shceme for a OU noise
"""

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt 
from matplotlib.animation import PillowWriter
import scipy.integrate as integrate
from Stochastic_Integral import SDEModel
import time
tim = time.time()

def V(x):
    return x**4/4 - x**2/2
def dV(x):
    return x**3 - x

def get_η_OU(N,dt):
    µ = lambda X,B,t: -X
    σ = lambda X,B,t: np.sqrt(2)
    SDE = SDEModel(µ,σ)
    return SDE.solve(N,dt,0,nb_simul=1)[0]
    
def SDEsolve(N_p,N,dt,θ,β,X_0,plot=True):
    """
    Solving the SDE of interacting particles
    N_p: number of particles
    N: number of steps
    dt: size of time step
    sigma, beta, X_0: parameters
    """
    #-----Init-----
    sqrt_dt = np.sqrt(dt)
    X = np.zeros((N_p,N+1))
    X[:,0] = X_0
    η_OU = get_η_OU(N,dt)

    def µ(X):
        return -(dV(X[:,n]) + θ*(X[:,n]-np.mean(X[:,n])))
    def σ():
        return np.sqrt(2/β)*η_OU[n]

    #-----Solver-----
    for n in range(N):
        #n is the time 
        X[:,n+1] = X[:,n] + µ(X)*dt + σ()*normal(0,sqrt_dt,N_p)

    #-----Plot-----
    #-----Fig-----
    if plot:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        ax1.set_title("Histogram") 
        ax2.set_title("Mean")
        writer1 = PillowWriter(fps=10)
        writer2 = PillowWriter(fps=10)

        with writer1.saving(fig1, f'histogram_β={β}_θ={θ}.gif', dpi=100):
            with writer2.saving(fig2, f'mean_β={β}_θ={θ}.gif', dpi=100):

                    for n in np.arange(0,N,10):
                        ax1.clear()
                        ax2.clear()
                        _, bins, _ = ax1.hist(X[:,n+1],100,density=True)
                        ax2.plot(range(n+1),np.mean(X[:,:n+1],axis=0))
                        writer1.grab_frame()
                        writer2.grab_frame()

    return X

def bifurcation_scheme():
    #PARAMETERS
    N = 70_000
    N_p = 2_000
    dt = 0.01
    θ = 1
    βs = np.linspace(1,4,50)
    ms = []
    for β in βs:
        X_0 = normal(0,np.sqrt(.1))
        ms.append(np.mean(SDEsolve(N_p,N,dt,θ,β,X_0,False)[-1]))

    plt.scatter(βs,ms)
    plt.show()
    return ms

if __name__=='__main__':

    #PARAMETERS
    N = 100_000
    N_p = 10_000
    β = 1
    dt = 0.01
    θ = 1
    ms = bifurcation_scheme()

