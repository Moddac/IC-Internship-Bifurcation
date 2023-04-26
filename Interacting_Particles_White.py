"""
Attempt to solve interacting particle SDE with Euler-Maruyama shceme for a white noise
"""

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt 
from matplotlib.animation import PillowWriter
import scipy.integrate as integrate
import time
tim = time.time()

def V(x):
    return x**4/4 - x**2/2
def dV(x):
    return x**3 - x

def pdf_ref(x,m):
    f = lambda y: np.exp(-β*(V(y)+.5*θ*(y-m)**2))
    try:
        Z
    except:
        Z = integrate.quad(f,-np.inf,np.inf)
    return f(x)/Z[0]

def E_2():
    return integrate.quad(lambda x: x**2*pdf_ref(x,0),-np.inf,np.inf)[0]

def θ_c():
    return 1/(β*E_2())

def SDEsolve(N_p,N,dt,θ,β,X_0,plot_ref=False):
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

    #-----Fig-----
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.set_title("Histogram") 
    ax2.set_title("Mean")
    writer1 = PillowWriter(fps=10)
    writer2 = PillowWriter(fps=10)

    def µ(X):
        return -(dV(X[:,n]) + θ*(X[:,n]-np.mean(X[:,n])))
    def σ():
        return np.sqrt(2/β)

    with writer1.saving(fig1, f'histogram_β={β}_θ={θ}.gif', dpi=100):
        with writer2.saving(fig2, f'mean_β={β}_θ={θ}.gif', dpi=100):

            for n in range(N):
                #n is the time 
                X[:,n+1] = X[:,n] + µ(X)*dt + σ()*normal(0,sqrt_dt,N_p)

                if n%10 == 0:
                    ax1.clear()
                    ax2.clear()
                    _, bins, _ = ax1.hist(X[:,n+1],100,density=True)
                    if plot_ref:
                        ax1.plot(bins,pdf_ref(bins,np.mean(X[:,n])))
                    ax2.plot(range(n+1),np.mean(X[:,:n+1],axis=0))
                    writer1.grab_frame()
                    writer2.grab_frame()

    return 


if __name__=='__main__':

    #PARAMETERS
    N = 10_000
    N_p = 1_000
    β = 2
    dt = 0.01
    θ = 1
    X_0 = normal(0,np.sqrt(.1)) 
    SDEsolve(N_p,N,dt,θ,β,X_0,True)

