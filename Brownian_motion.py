"""
Creation of a Brownian walk & plot
"""

import numpy as np 
from numpy.random import normal
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import time 

def simul_walk(T,N):
    """
    Simulate the Brownian walk, T is the final time & N is the number of steps 
    """
    #-----Init----
    h = T/N
    ts = np.arange(0,T,h)
    W = 0
    Ws = []

    for t in ts:
        δ = normal(0,h)
        W = W + δ 
        Ws.append(W)

    return np.array(Ws)

def plot_walk(T,N):
    """
    plotting the Brownian walk 
    """
    h = T/N
    ts = np.arange(0,T,h)
    Ws = simul_walk(T,N)

    plt.plot(ts,Ws)
    plt.plot((0,T),(0,0))
    plt.show()
    return Ws

def gif_walk(T,N,number_walks=1):
    """
    Animating the walk 
    """
    #-----Init----
    h = T/N
    ts = np.arange(0,T,h)
    W_last = np.zeros(number_walks)
    W_new = np.zeros(number_walks)

    fig, ax = plt.subplots()
    ax.set_xlim(0,T)
    ax.set_ylim(-3,3)
    ax.plot((0,T),(0,0),color='orange')
    ax.set_title(f'Brownian walks for T={T} and N={N}')
    writer = PillowWriter(fps=15)

    with writer.saving(fig, 'brownian_walk.gif', dpi=100):
        for i,t in enumerate(ts[:-1]):
            for n in range(number_walks):
                δ = normal(0,h)
                W_new[n] = W_last[n] + δ  
                ax.plot(ts[i:i+2],[W_last[n],W_new[n]],color=f'{n/number_walks}')
                W_last[n] = W_new[n]

            writer.grab_frame()
    
    return        

def anim_walk(T,N,number_walks=1):
    """
    Movie of the walk
    """
    #-----Init----
    # to run GUI event loop
    plt.ion()
    h = T/N
    ts = np.arange(0,T,h)
    W_last = np.zeros(number_walks)
    W_new = np.zeros(number_walks)

    fig, ax = plt.subplots()
    ax.plot((0,T),(0,0),color='orange')
    ax.set_title(f'Brownian walks for T={T} and N={N}')

    for i,t in enumerate(ts[:-1]):
        for n in range(number_walks):
            δ = normal(0,h)
            W_new[n] = W_last[n] + δ  
            ax.plot(ts[i:i+2],[W_last[n],W_new[n]],color=f'{n/number_walks}')
            ax.set_xlim(0,t)
            W_last[n] = W_new[n]

            # drawing updated values
            fig.canvas.draw()
        
            # This will run the GUI event
            # loop until all UI events
            # currently waiting have been processed
            fig.canvas.flush_events()
        
            # time.sleep(0.001)

    
if __name__=='__main__':
    T = 10
    N = 300
    # plot_walk(T,N)
    gif_walk(T,N,number_walks=3)

