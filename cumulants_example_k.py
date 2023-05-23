#%%
"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
By trying to solve the equation on the cumulants k_n and without being stationnary 
"""

import numpy as np
from numpy.random import normal
from sympy import bell
from math import factorial, comb
from scipy.optimize import root, newton
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

ν = .5


def f(t, k, α, θ, σ_m, σ):
    """
    This is the function to solve dk/dt = f(k,t) 
    with k being a vector of function at a fix time t: k(t) = (k_1(t);...;k_N(t)) 
    """
    # -----Init-----
    N = k.shape[0]
    _k = np.insert(k, [0, N, N], [0, 0, 0])

    # F is define with one more element because equations in the paper
    # Starts at n=1, so we will keep it that way
    F = np.zeros(N)
    F[0] = θ*_k[1]
    F[1] = σ**2 #it is not sigma² / 2 because we multiply by n=2

    # -----Definition of f-----
    """
    To define f a sum from 1 to n-1 which contains the three sums will be computed
    Then the last terms of the sum are added
    And we suppose B.C.: k_{N+1} = k_{N+2} = 0
    """
    for n in range(1, N+1):
        Σ = np.sum([
            σ_m**2/2*_k[i]*_k[n-i] / (factorial(i-1)*factorial(n-i-1))
            - 3*_k[i]*_k[n-i+2] / (factorial(i-1)*factorial(n-i))
            - np.sum([
                _k[i]*_k[j]*_k[n+2-i-j] /
                (factorial(i-1)*factorial(j-1)*factorial(n-i-j+1))
                for j in range(1, n-i+2)])
            for i in range(1, n)])

        F[n-1] += n*((α-θ+σ_m**2*(ν+(n-1)/2))*_k[n]
                   - _k[n+2]
                   + factorial(n-1)*(
                    Σ
                   - 3*_k[n]*_k[2]/factorial(n-1)
                   - _k[n]*_k[1]**2/factorial(n-1)))

    return F


def SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ):
    """
    Solving the cumulant ODE for 1 set of parameters
    """
    # -----Init-----
    k0 = np.zeros(N)
    k0[0] = 1; k0[1] = 1.5**2

    # -----Solver-----
    k = solve_ivp(f, (t0, t_end), k0,
                  args=(α, θ, σ_m, σ), method='Radau')

    return k

def SolveCumulant_Stationnary(N,α,θ,σ_m,σ):
    """
    Solving the cumulant ODE for 1 set of parameters
    And in stationnary state
    """
    # -----Init-----
    k0 = 0.01*np.ones(N)
    k0[0] = .5 ; k0[1] = .5


    # -----Solver-----
    g = lambda x: f(0,x,α,θ,σ_m,σ)
    k = root(g, k0)

    return k

#%%
if __name__ == '__main__':

    # -----Init-----
    Ns = [4,8,16]
    α = 1
    θ = 4
    σ_m = .8
    σs = list(np.linspace(1.8, 1.892, 100)) + list(np.linspace(1.893,2.,100))
    σs = np.linspace(1.8,2.,200)

    t0 = 0
    t_end = 10e4

    # -----Solving-----
    print("##########")
    print("Starting solving...")
    for N in Ns:
        print(f"Solving for N={N}...")
        M1s = []

        for i,σ in enumerate(σs):
            M1 = SolveCumulant_ODE(N, t0, t_end, α, θ, σ_m, σ)
            M1s.append(M1.y[0,-1])
            
            # M1 = SolveCumulant_Stationnary(N, α, θ, σ_m, σ)
            # M1s.append(M1.x[0])

        # -----Plot-----
        plt.scatter(σs, M1s,label=f"N={N}")

    plt.xlabel("σ")
    plt.ylabel("m")
    plt.legend()
    plt.title("Mean with cumulant method")
    plt.show()



# %%
