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

def get_η_OU(N, dt):
    def µ(X, B, t): return -X
    def σ(X, B, t): return np.sqrt(2)
    SDE = SDEModel(µ, σ)
    return SDE.solve(N, dt, 0, nb_simul=1)[0]


def SDEsolve(N_p, N, dt, θ, β, X_0):
    """
    Solving the SDE of interacting particles
    N_p: number of particles
    N: number of steps
    dt: size of time step
    theta, beta, X_0: parameters
    """
    # -----Init-----
    sqrt_dt = np.sqrt(dt)
    X = np.zeros((N_p, N+1))
    X[:, 0] = X_0
    η_OU = get_η_OU(N, dt)

    def µ(X):
        return -(dV(X[:, n]) + θ*(X[:, n]-np.mean(X[:, n])))

    def σ():
        return np.sqrt(2/β)*η_OU[n]

    # -----Solver-----
    for n in range(N):
        # n is the time
        X[:, n+1] = X[:, n] + µ(X)*dt + σ()*normal(0, sqrt_dt, N_p)

    return X


def bifurcation_scheme():
    # PARAMETERS
    N = 70_000
    N_p = 2_000
    dt = 0.01
    θ = 1
    βs = np.linspace(1, 4, 50)
    ms = []
    for β in βs:
        X_0 = normal(0, np.sqrt(.1))
        ms.append(np.mean(SDEsolve(N_p, N, dt, θ, β, X_0)[-1]))

    plt.scatter(βs, ms)
    plt.show()
    return ms


if __name__ == '__main__':

    # PARAMETERS
    N = 100_000
    N_p = 10_000
    β = 1
    dt = 0.01
    θ = 1
    ms = bifurcation_scheme()
