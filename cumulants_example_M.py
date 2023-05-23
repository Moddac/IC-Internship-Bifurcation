"""
Example of cumulant method as shown in "Dimension reduction of noisy interacting systems"
"""

import numpy as np
from numpy.random import normal
from sympy import bell
from math import factorial, comb
from scipy.optimize import fsolve, newton
import matplotlib.pyplot as plt


def Σ(x,σ,σ_m):
    return σ**2 + σ_m**2*x**2

def V_α(x,α):
    return x**4/4 - α*x**2/2

def B(n,l,M):
    return bell(n,l,M[1:])

def f(M1,M2,N,α,θ,σ_m,σ):
    M = np.zeros(N+4,dtype=np.float64)
    M[-1] = 0
    M[0] = 1
    M[1] = M1
    M[2] = M2

    for n in range(1,N+1):
        # α_r = α + σ_m*normal(0,.1)
        M[n+2] = n*(α - θ + n/2*σ_m**2)*M[n] + n*(n-1)/2*σ**2*M[n-2] + θ*M[1]*M[n-1]

    return M

def cummulant(M1,M2,N,α,θ,σ_m,σ):

    M = f(M1,M2,N,α,θ,σ_m,σ)
    M_n1 = 0

    for l in range(2,N+2):
        M_n1 -= (-1)**(l-1)*factorial(l-1)*B(N+1,l,M)

    M_n2 = (N+2)*M_n1*M[1]

    for k in range(2,N+1):
        M_n2 += .5*comb(N+2,k)*M[k]*M[N+2-k]    
    for l in range(3,N+3):
        M_n2 -= (-1)**(l-1)*factorial(l-1)*B(N+2,l,M)

    return np.array([M_n1,M_n2],dtype=np.float64)

def bifuraction(N,α,θ,σ_m):

    σs = np.linspace(.5,2.5,100)
    M1s = []
    for σ in σs:
        print(f'σ={σ}')
        x0 = 2*[np.sqrt((2-σ)/1.5) if σ<2 else 0]
        try:
            root = fsolve(lambda M: f(M[0],M[1],N,α,θ,σ_m,σ)[[N+1,N+2]]-cummulant(M[0],M[1],N,α,θ,σ_m,σ),x0=[σ,σ])
        except RuntimeError:
            root = [0,0]
        M1s.append(root[0])

    plt.scatter(σs,M1s)
    plt.show()
    
if __name__=='__main__':

    N = 4
    α = 1
    θ = 4
    σ_m = .2
    bifuraction(N,α,θ,σ_m)
